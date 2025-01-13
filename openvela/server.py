# openvela/server.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .completion import run_completion
from .runner import run_workflow

app = FastAPI()


class WorkflowRequest(BaseModel):
    provider: str
    workflow_type: str
    base_url_or_api_key: str
    model: str
    options: dict = {}
    agents: list = []
    task: str


class CompletionRequest(BaseModel):
    provider: str
    base_url_or_api_key: str
    model: str
    options: dict = {}
    messages: list = []


@app.post("/run_workflow")
async def run_workflow_endpoint(request_data: WorkflowRequest):
    try:
        # Use dot notation to access attributes of request_data instead of dictionary-like access
        worflow_response = run_workflow(
            {
                "provider": request_data.provider,
                "workflow_type": request_data.workflow_type,
                "base_url_or_api_key": request_data.base_url_or_api_key,
                "model": request_data.model,
                "options": request_data.options,
                "agents_definitions": request_data.agents,
                "task_description": request_data.task,
            }
        )
        return worflow_response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/completion")
async def completion_endpoint(request_data: CompletionRequest):
    try:
        completion_response = run_completion(
            {
                "provider": request_data.provider,
                "base_url_or_api_key": request_data.base_url_or_api_key,
                "model": request_data.model,
                "options": request_data.options,
                "messages": request_data.messages,
            }
        )
        return completion_response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def run_server(host="0.0.0.0", port=8000):
    uvicorn.run("openvela.server:app", host=host, port=port, reload=False)
