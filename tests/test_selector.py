import logging

from openvela.agents import Agent, SupervisorAgent
from openvela.llms import GroqModel
from openvela.tasks import Task
from openvela.workflows import AutoSelectWorkflow

# Make sure you have installed openvela before running:
# pip install openvela


# Configure logging for visibility
logging.basicConfig(level=logging.INFO)

# Initialize the GroqModel with your valid API key
model_instance = GroqModel(
    api_key="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"
)

# ---------------------------------------------------------------------------------
# 1. Define New Agents
# ---------------------------------------------------------------------------------

designer_agent = Agent(
    settings={
        "name": "DesignerAgent",
        "prompt": (
            "You are the DesignerAgent. Your task is to provide a visually appealing "
            "and user-centric design for a website with an API. You will: \n\n"
            "1. Suggest a style guide (colors, fonts, layout). \n"
            "2. Outline basic wireframes and layout structure. \n"
            "3. Provide any relevant CSS and design documentation needed. \n\n"
            "Please output your best possible HTML/CSS suggestions or design guidelines."
        ),
        "description": "Generate UI/UX design elements for the website.",
    },
    model=model_instance,
)

front_end_developer_agent = Agent(
    settings={
        "name": "FrontEndDeveloperAgent",
        "prompt": (
            "You are the FrontEndDeveloperAgent. Your job is to build the front end of "
            "a website based on the DesignerAgent's suggestions. You should:\n\n"
            "1. Produce HTML, CSS, and optionally JavaScript code. \n"
            "2. Implement a user-friendly interface that follows the design guidelines.\n"
            "3. Integrate calls to the backend API endpoints (which the BackEndDeveloperAgent provides) "
            "using JavaScript fetch or a similar mechanism.\n\n"
            "Please provide a single HTML file (and inline or separate CSS/JS) that can "
            "be used to render the interface."
        ),
        "description": "Build the front end of the website following the design specifications.",
    },
    model=model_instance,
)

back_end_developer_agent = Agent(
    settings={
        "name": "BackEndDeveloperAgent",
        "prompt": (
            "You are the BackEndDeveloperAgent. Your role is to implement a robust backend "
            "with a simple API. You should:\n\n"
            "1. Use Python (Flask) for simplicity in this demonstration. \n"
            "2. Create at least one GET and one POST endpoint, illustrating how the front end "
            "would integrate with them. \n"
            "3. Provide instructions on how to run this backend.\n\n"
            "Your output should include a Python file (or code snippet) that starts a Flask app."
        ),
        "description": "Implement the backend service and API endpoints for the website.",
    },
    model=model_instance,
)

# A supervisor agent to oversee or select the best outputs from each specialized agent
supervisor = SupervisorAgent(
    settings={
        "name": "SupervisorAgent",
        "prompt": (
            "You are the SupervisorAgent. You will coordinate the DesignerAgent, "
            "FrontEndDeveloperAgent, and BackEndDeveloperAgent to produce a fully functional "
            "website with an API. Ensure:\n\n"
            "1. The final solution has coherent design (DesignerAgent), front end (FrontEndDeveloperAgent), "
            "and back end (BackEndDeveloperAgent).\n"
            "2. The final code is presented cleanly.\n"
            "3. All instructions for running the application are clear.\n"
            "4. If improvements are needed, ask for clarifications or refinements.\n\n"
            "Finally, consolidate the outputs into one cohesive result."
        ),
    },
    agent_type="selector",
    model=model_instance,
)

# ---------------------------------------------------------------------------------
# 2. Define the Task
# ---------------------------------------------------------------------------------
task = Task(
    prompt=(
        "Your objective is to collaborate on building a complete website with a simple Flask API. "
        "We need:\n\n"
        "1. A user-centric design and style guide (DesignerAgent). \n"
        "2. A functional front end that leverages the design and consumes the API (FrontEndDeveloperAgent). \n"
        "3. A backend API in Python Flask, demonstrating at least one GET and one POST endpoint "
        "(BackEndDeveloperAgent). \n\n"
        "Finally, combine everything into a single or consolidated multi-file code snippet that a user "
        "can run locally to see the site in action.\n\n"
        "Output the final code (HTML, CSS, JavaScript, and Python)."
    ),
    agents=[designer_agent, front_end_developer_agent, back_end_developer_agent],
)

# ---------------------------------------------------------------------------------
# 3. Create an AutoSelectWorkflow with validation if desired
# ---------------------------------------------------------------------------------
workflow = AutoSelectWorkflow(
    task=task,
    agents=[designer_agent, front_end_developer_agent, back_end_developer_agent],
    supervisor=supervisor,
    validate_output=True,  # You can set this to False if you don't want validation
    max_attempts=3,  # Number of attempts the workflow will try if validation fails
)

# ---------------------------------------------------------------------------------
# 4. Run the Workflow and Print the Final Output
# ---------------------------------------------------------------------------------
final_output = workflow.run()
print("Final Output:\n", final_output)
