"""Basic agent implementations."""

from dataclasses import dataclass
from typing import Dict, Any

from .base import BaseAgent
from ..core import AgentConfig, AgentType


@dataclass
class Agent(BaseAgent):
    """Standard agent implementation."""
    
    def get_agent_type(self) -> str:
        """Return the agent type identifier."""
        return AgentType.SIMPLE


@dataclass  
class StartAgent(BaseAgent):
    """Agent that initiates workflows."""
    
    def get_agent_type(self) -> str:
        """Return the agent type identifier.""" 
        return "start"
    
    def _build_user_input(self) -> str:
        """Build user input with start-agent specific logic."""
        base_input = super()._build_user_input()
        
        # Add workflow initiation context
        if base_input:
            return f"[WORKFLOW START] {base_input}"
        return "[WORKFLOW START] Begin the workflow with your initial response."


@dataclass
class EndAgent(BaseAgent):
    """Agent that concludes workflows."""
    
    def get_agent_type(self) -> str:
        """Return the agent type identifier."""
        return "end"
    
    def _build_user_input(self) -> str:
        """Build user input with end-agent specific logic."""
        base_input = super()._build_user_input()
        
        # Add workflow conclusion context
        if base_input:
            return f"[WORKFLOW END] {base_input}"
        return "[WORKFLOW END] Provide the final synthesis and conclusion."
    
    def generate(self, **kwargs: Any) -> str:
        """Generate final response and mark workflow completion."""
        response = super().generate(**kwargs)
        
        # Log workflow completion
        self.logger.info(f"Workflow completed by {self.name}")
        
        return response