
description: str = "You are AI agent capable analysing the request, answer or use external tools in case think is reasonable"

json_template: str = """
  When providing the answer use following JSON format:

{
    "answered": bool,
    "tool": string,
    "output": string
}
Where the variables are:
- answered: true if the output is an answer for the question, false if the output will be used by a tool
- tool: empty if no tool required or search if you need to search the web, or scrape if you need to scrape a website
- output: the answer of the question, or the request to be send to the tool.
"""

tools: str = """
Available Tools:
- search: search the web returning a list of links and descriptions
- scrape: download the content of a website in text format

"""

template: str = f"""
{description}

{json_template}

{tools}

"""
