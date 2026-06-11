from typing import TypedDict
import data
import yfinance as fin
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from groq import Groq
import pandas as pd

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

class AgentState(TypedDict):
    ticker: str
    fundamentals: dict
    ten_k_summary:str
    risk_info: dict
    info_collected: dict
    all_agent_done:list
    final_output: str


def fundamentals_agent(state:AgentState):
    fundamentals_dict = data.getVitals(state['ticker'])
    status = state['all_agent_done']
    ten_k_text = data.get_latest_10k(state['ticker'])
    slices = data.slice_10k_specific_items(ten_k_text)

    return {
        "fundamentals" : {
            **fundamentals_dict,
            "ten_k": {
                "item 1":slices['Item 1'],
                "item 1a":slices['Item 1A'],
                "item 7":slices['Item 7'],
                "item 8":slices['Item 8']
            }
        },
        #"all_agent_done" : status.append('Fundamentals agent done')
    }


def risk_agent(state:AgentState):
    fundam = state['fundamentals']
    price_hist = pd.Series(fundam['priceHist'])  # convert list back to series so pandas math methods work
    daily_returns = pd.Series(fundam['dailyReturns'])

    status = state['all_agent_done']

    volatility = daily_returns.std()
    beta = fundam['beta']
    rolling_max = price_hist.expanding().max()
    drawdown = (price_hist - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    return {
        "risk_info":{
            "volatility" : float(volatility),
            "beta": beta,
            "max_drawdown" : max_drawdown
        },
        #"all_agent_done" : status.append('Risk agent done')
    }


def info_agent(state:AgentState):
    ticker = state['ticker']
    status = state['all_agent_done']
    insider_trans = fin.Ticker(ticker).insider_transactions.astype(str)
    return {
        "info_collected":{
            "insider_trans" : insider_trans.to_dict(orient='records')
        },
        #"all_agent_done" : status.append('Info agent done')
    }

#----------------------------

#----------------------------
# i want to make a summarising agent that extracts the imp info from the 10k and addes it to the state

def summary_agent(state:AgentState):
    """Summarise the 10k financial form and keep only useful information"""

    chunks_dict = state["fundamentals"]['ten_k']

    status = state['all_agent_done']

    section_prompts = {
    "item 1": "Extract a concise Business Overview from this section. Cover: what the company does, key products and services, revenue segments, and geographic markets.",  # business context
    
    "item 1a": "Extract the top 5 most material risk factors from this section. One sentence per risk, as close to verbatim as possible. Focus on risks that could materially impact financial performance.",  # risk context
    
    "item 7": "Extract key insights from Management Discussion and Analysis. Cover: revenue trends, margin commentary, management's outlook, and any forward guidance explicitly stated.",  # management context
    
    "item 8": "Extract key financial highlights from this section. Cover: total revenue, net income, gross margin, YoY growth rates, and any significant balance sheet items mentioned."  # financial context
    }

    system_prompt = """
    You are a senior financial analyst extracting key information from a specific section of a 10-K annual report.

    Extract ONLY what is explicitly stated in the text provided. Do not infer, speculate, or add context not present in the filing.

    Rules:
    - If the information requested is not found in the text, write "Insufficient data in extract"
    - Do not add analysis or opinion — extraction only
    - No legal boilerplate, exhibit lists, or SEC filing metadata
    - Be concise — maximum 150 words per response
    - Do not repeat the section name or instructions back
    """ 

    summaries = {}

    for section_name, chunk_text in chunks_dict.items():
        if chunk_text is None:
            summaries[section_name] = "Insufficient data"  
            continue


        user_prompt = f"{section_prompts[section_name]} Section text: {chunk_text[:30000]}"  #capped becuase of groq limits :((


        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        summaries[section_name] = response.choices[0].message.content

    compiled = "\n\n".join([f"{k}:\n{v}" for k, v in summaries.items()])

    return {"ten_k_summary": compiled,
            #"all_agent_done" : status.append('Summary agent done')
            }  
    


def synth_agent(state:AgentState):
    """Make a summary kind of something from the data in the state and write to final_output"""

    user_prompt = f"""
    Ticker: {state['ticker']}
    Sector: {state['fundamentals']['sector']}
    P/E Ratio: {state['fundamentals']['peRatio']}
    Beta: {state['risk_info']['beta']}
    Volatility: {state['risk_info']['volatility']}
    Max Drawdown: {state['risk_info']['max_drawdown']}
    Insider Transactions: {state['info_collected']['insider_trans']}
    10-K Summary: {state['ten_k_summary']}
    """

    system_prompt = """
    You are a senior financial analyst at a PE firm. You are concise, direct, and data-driven. No filler sentences.

    Given company data, produce a due diligence brief in exactly this structure:

    1. Company Snapshot — ticker, sector, key metrics formatted correctly (P/E as ratio, volatility as %, drawdown as %)
    2. Financial Analysis — what the numbers indicate about the company's health
    3. Risk Assessment — beta, volatility, drawdown interpreted in context
    4. Insider Activity — what the trading patterns suggest
    5. Bull Case — 2-3 specific reasons to invest
    6. Bear Case — 2-3 specific reasons to avoid
    7. Recommendation — Buy / Hold / Sell with one sentence justification

    If the 10-K extract is insufficient to draw conclusions, say so explicitly. Do not fabricate insights.
    """

    status = state['all_agent_done']

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    #result = response.choices[0].message.content

    return {
    "final_output": response.choices[0].message.content,
    #"all_agent_done" : status.append('Synth agent done')
    }

# test_state = {"ticker": "AAPL", "fundamentals": {}, "risk_info": {}, "info_collected": {}, "all_agent_done": [], "final_output": ""}
# print(fundamentals_agent(test_state))


graph = StateGraph(AgentState)

graph.add_node("fundamentals_agent",fundamentals_agent)
graph.add_node("risk_agent",risk_agent)
graph.add_node("info_agent",info_agent)
graph.add_node("synth_agent",synth_agent)
graph.add_node("summary_agent",summary_agent)

graph.add_edge(START, "fundamentals_agent")

graph.add_edge("fundamentals_agent", "summary_agent")
graph.add_edge("fundamentals_agent", "risk_agent")
graph.add_edge("fundamentals_agent", "info_agent")

graph.add_edge("risk_agent", "synth_agent")
graph.add_edge("info_agent", "synth_agent")
graph.add_edge("summary_agent", "synth_agent")

graph.add_edge("synth_agent", END)


app = graph.compile()

result = app.invoke({"ticker": "AAPL", "fundamentals": {}, "risk_info": {}, "info_collected": {}, "all_agent_done": [], "final_output": ""})

print(result.keys())