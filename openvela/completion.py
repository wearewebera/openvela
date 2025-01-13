import sys

from .llms import GroqModel, OllamaModel, OpenAIModel


def run_completion(config):
    provider = config["provider"]
    base_url_or_api_key = config["base_url_or_api_key"]
    model_name = config["model"]
    options = config["options"]
    messages = config["messages"]
    print(config)
    output = ""
    # Initialize the model instance based on the provider
    if provider == "openai":
        model_instance = OpenAIModel(api_key=base_url_or_api_key, model=model_name)
        output = model_instance.generate_response(messages, options)
    elif provider == "groq":
        model_instance = GroqModel(api_key=base_url_or_api_key, model=model_name)
        output = model_instance.generate_response(messages, options)
    elif provider == "ollama":
        model_instance = OllamaModel(base_url=base_url_or_api_key, model=model_name)
        output = model_instance.generate_response(messages, options)
    else:
        print(f"Unsupported provider: {provider}")
        sys.exit(1)

    completion_response = {
        "provider": provider,
        "model": model_name,
        "messages": messages,
        "output": output,
    }
    # Call the completion function of the model instance

    return completion_response
