from typing import TypedDict

class AgentState(TypedDict):
    ticker: str
    fundaments: dict
    risk_info: dict
    info_collected: dict
    all_agent_done:list
    final_output: str






