from fastapi import FastAPI, WebSocket
from agent import app
from fastapi.middleware.cors import CORSMiddleware


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.websocket('/test')
async def eval_model(websocket: WebSocket):
    # Accept the incoming WebSocket connection from the client
    # Without this the connection is rejected
    await websocket.accept()
    
    # Wait for the client to send a ticker string e.g. "AAPL"
    ticker = await websocket.receive_text()
    
    # Build the initial state to pass into the LangGraph graph
    # All agent outputs start empty — graph will fill them in
    initial_state = {
        "ticker": ticker,
        "fundamentals": {},
        "risk_info": {},
        "info_collected": {},
        "all_agent_done": [],
        "final_output": ""
    }
    
    # app.stream() runs the graph and yields one chunk per agent completion
    # Instead of waiting for everything, we get updates as each agent finishes
    for chunk in app.stream(initial_state):
        # Send each agent's output to the frontend as JSON
        # Frontend receives these one by one as agents complete
        await websocket.send_json(chunk)
    
    # All agents done, close the connection cleanly
    await websocket.close()