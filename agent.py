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
    risk_info: dict
    info_collected: dict
    all_agent_done:list
    final_output: str


def fundamentals_agent(state:AgentState):
    fundamentals_dict = data.getVitals(state['ticker'])

    ten_k_text = data.get_latest_10k(state['ticker'])

    return {
        "fundamentals" : {
            **fundamentals_dict,
            "ten_k": ten_k_text
        }
    }


def risk_agent(state:AgentState):
    fundam = state['fundamentals']
    price_hist = pd.Series(fundam['priceHist'])  # convert list back to series so pandas math methods work
    daily_returns = pd.Series(fundam['dailyReturns'])

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
        }
    }


def info_agent(state:AgentState):
    ticker = state['ticker']
    insider_trans = fin.Ticker(ticker).insider_transactions.astype(str)
    return {
        "info_collected":{
            "insider_trans" : insider_trans.to_dict(orient='records')
        }
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
    10-K Extract: {state['fundamentals']['ten_k'][:3000]}
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

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    #result = response.choices[0].message.content

    return {
    "final_output": response.choices[0].message.content
    }

# test_state = {"ticker": "AAPL", "fundamentals": {}, "risk_info": {}, "info_collected": {}, "all_agent_done": [], "final_output": ""}
# print(fundamentals_agent(test_state))


graph = StateGraph(AgentState)

graph.add_node("fundamentals_agent",fundamentals_agent)
graph.add_node("risk_agent",risk_agent)
graph.add_node("info_agent",info_agent)
graph.add_node("synth_agent",synth_agent)

graph.add_edge(START, "fundamentals_agent")

graph.add_edge("fundamentals_agent", "risk_agent")
graph.add_edge("fundamentals_agent", "info_agent")

graph.add_edge("risk_agent", "synth_agent")
graph.add_edge("info_agent", "synth_agent")

graph.add_edge("synth_agent", END)


app = graph.compile()

#result = app.invoke({"ticker": "AAPL", "fundamentals": {}, "risk_info": {}, "info_collected": {}, "all_agent_done": [], "final_output": ""})

