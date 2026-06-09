from typing import TypedDict
import data
import yfinance as fin
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = Groq(api_key=api_key)

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
    volatility = fundam['dailyReturns'].std()
    beta = fundam['beta']
    rolling_max = fundam['priceHist'].expanding().max()
    drawdown = (fundam['priceHist'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "risk_info":{
            "volatility" : volatility,
            "beta": beta,
            "max_drawdown" : max_drawdown
        }
    }


def info_agent(state:AgentState):
    ticker = state['ticker']
    insider_trans = fin.Ticker(ticker).insider_transactions
    return {
        "info_collected":{
            "insider_trans" : insider_trans
        }
    }

def synth_agent(state:AgentState):
    """Make a summary kind of something from the data in the state and write to final_output"""

    return {
    "final_output": f"Ticker: {state['ticker']} | Sector: {state['fundamentals']['sector']} | Beta: {state['risk_info']['beta']}"
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

# result = app.invoke({"ticker": "AAPL", "fundamentals": {}, "risk_info": {}, "info_collected": {}, "all_agent_done": [], "final_output": ""})
# print(result['final_output'])


