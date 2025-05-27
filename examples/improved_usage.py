"""
Example demonstrating the improved, more Pythonic OpenVela patterns.

This example shows how to use the new type-safe configuration,
factory patterns, and better error handling.
"""

import logging
from pathlib import Path

from openvela import (
    # New type-safe configuration
    AgentConfig,
    WorkflowConfig,
    ModelConfig,
    # Enums for type safety
    AgentType,
    WorkflowType,
    ModelProvider,
    # Factories for clean object creation
    AgentFactory,
    WorkflowFactory,
    ModelFactory,
    auto_register_factories,
    # Improved exception handling
    OpenVelaError,
    ConfigurationError,
    AgentError,
)

# Initialize factories
auto_register_factories()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_type_safe_workflow():
    """Demonstrate creating a workflow with type-safe configuration."""
    
    try:
        # Create model configuration with validation
        model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            api_key="your-api-key-here",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create workflow configuration with validation
        workflow_config = WorkflowConfig(
            type=WorkflowType.CHAIN_OF_THOUGHT,
            validate_output=True,
            max_attempts=3,
            max_previous_messages=10
        )
        
        # Create agent configurations with full validation
        start_agent_config = AgentConfig(
            name="StartAgent",
            prompt="You are the StartAgent. Begin by outlining the main objectives.",
            input="What are the primary goals for this task?",
            agent_type=AgentType.SIMPLE
        )
        
        middle_agent_config = AgentConfig(
            name="AnalysisAgent", 
            prompt="You are the AnalysisAgent. Provide detailed analysis of the objectives.",
            input="Analyze the objectives in depth.",
            agent_type=AgentType.SIMPLE,
            options={"temperature": 0.5}  # Agent-specific options
        )
        
        end_agent_config = AgentConfig(
            name="EndAgent",
            prompt="You are the EndAgent. Synthesize all information into a final answer.",
            input="Provide the final synthesis.",
            agent_type=AgentType.SIMPLE
        )
        
        # Create model instance using factory
        model = ModelFactory.create_model(model_config)
        
        # Create agents using factory with automatic validation
        start_agent = AgentFactory.create_agent(start_agent_config, model)
        middle_agent = AgentFactory.create_agent(middle_agent_config, model)
        end_agent = AgentFactory.create_agent(end_agent_config, model)
        
        logger.info(f"Created agents: {start_agent.name}, {middle_agent.name}, {end_agent.name}")
        
        # Create supervisor (could also use factory)
        supervisor_config = AgentConfig(
            name="SupervisorAgent",
            prompt="Oversee the workflow and select the next agent.",
            agent_type=AgentType.SELECTOR  # Use enum for type safety
        )
        supervisor = AgentFactory.create_agent(supervisor_config, model)
        
        # Create workflow using factory
        workflow = WorkflowFactory.create_workflow(
            workflow_config,
            agents=[middle_agent],
            supervisor=supervisor,
            start_agent=start_agent,
            end_agent=end_agent,
            task_prompt="Create a comprehensive business strategy for a tech startup"
        )
        
        logger.info(f"Created workflow: {workflow_config.type}")
        
        # Execute workflow with proper error handling
        result, memory_id = workflow.run()
        
        logger.info(f"Workflow completed successfully. Memory ID: {memory_id}")
        return result
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        if hasattr(e, 'field'):
            logger.error(f"Invalid field: {e.field} = {e.value}")
        raise
    except AgentError as e:
        logger.error(f"Agent error in {e.agent_name}: {e}")
        raise
    except OpenVelaError as e:
        logger.error(f"OpenVela error: {e}")
        if e.details:
            logger.error(f"Details: {e.details}")
        raise


def create_workflow_from_json():
    """Demonstrate creating workflow from JSON configuration."""
    
    # Example JSON configuration (could be loaded from file)
    config_data = {
        "workflow": {
            "type": "cot",
            "validate_output": True,
            "max_attempts": 3
        },
        "model": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "your-api-key-here",
            "temperature": 0.7
        },
        "agents": [
            {
                "name": "StartAgent",
                "prompt": "Begin the task analysis",
                "input": "What should we focus on?",
                "agent_type": "simple"
            },
            {
                "name": "ResearchAgent",
                "prompt": "Conduct thorough research",
                "input": "Research the topic thoroughly",
                "agent_type": "simple"
            },
            {
                "name": "EndAgent",
                "prompt": "Provide final conclusions",
                "input": "Synthesize all findings",
                "agent_type": "simple"
            }
        ]
    }
    
    try:
        # Create configurations from dictionaries (with validation)
        workflow_config = WorkflowConfig(**config_data["workflow"])
        model_config = ModelConfig(**config_data["model"])
        
        # Create model
        model = ModelFactory.create_model(model_config)
        
        # Create agents from dictionary configurations
        agents = []
        for agent_data in config_data["agents"]:
            agent = AgentFactory.create_from_dict(agent_data, model)
            agents.append(agent)
        
        # Create supervisor from first agent (simplified example)
        supervisor = agents[0]
        start_agent = agents[0] if len(agents) > 0 else None
        end_agent = agents[-1] if len(agents) > 2 else None
        middle_agents = agents[1:-1] if len(agents) > 2 else []
        
        # Create workflow
        workflow = WorkflowFactory.create_workflow(
            workflow_config,
            agents=middle_agents,
            supervisor=supervisor,
            start_agent=start_agent,
            end_agent=end_agent,
            task_prompt="Analyze market trends for emerging technologies"
        )
        
        # Execute
        result, memory_id = workflow.run()
        
        logger.info("JSON-configured workflow completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to create workflow from JSON: {e}")
        raise


def demonstrate_validation():
    """Demonstrate configuration validation catching errors early."""
    
    try:
        # This will raise a validation error - empty name
        AgentConfig(
            name="",  # Invalid: empty name
            prompt="Test prompt"
        )
    except ValueError as e:
        logger.info(f"Caught validation error as expected: {e}")
    
    try:
        # This will raise a validation error - invalid temperature
        ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=3.0  # Invalid: > 2.0
        )
    except ValueError as e:
        logger.info(f"Caught temperature validation error as expected: {e}")
    
    try:
        # This will raise a validation error - invalid max_attempts
        WorkflowConfig(
            type=WorkflowType.CHAIN_OF_THOUGHT,
            max_attempts=0  # Invalid: must be >= 1
        )
    except ValueError as e:
        logger.info(f"Caught max_attempts validation error as expected: {e}")


if __name__ == "__main__":
    logger.info("=== OpenVela Improved Patterns Demo ===")
    
    # Demonstrate validation
    logger.info("\n1. Testing configuration validation...")
    demonstrate_validation()
    
    # Demonstrate type-safe workflow creation
    logger.info("\n2. Creating type-safe workflow...")
    try:
        result1 = create_type_safe_workflow()
        logger.info("Type-safe workflow demo completed")
    except Exception as e:
        logger.error(f"Type-safe workflow failed: {e}")
    
    # Demonstrate JSON configuration
    logger.info("\n3. Creating workflow from JSON...")
    try:
        result2 = create_workflow_from_json()
        logger.info("JSON workflow demo completed")
    except Exception as e:
        logger.error(f"JSON workflow failed: {e}")
    
    logger.info("\n=== Demo completed ===")