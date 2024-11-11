# openvela/cli.py

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="OpenVela CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # openvela serve
    parser_serve = subparsers.add_parser("serve", help="Run the OpenVela API server")
    parser_serve.add_argument(
        "--host", default="0.0.0.0", help="Host to run the server on"
    )
    parser_serve.add_argument(
        "--port", default=8000, type=int, help="Port to run the server on"
    )

    # openvela interface
    parser_interface = subparsers.add_parser(
        "interface", help="Run the OpenVela interface"
    )

    # openvela run
    parser_run = subparsers.add_parser(
        "run", help="Run a workflow based on command-line arguments"
    )
    parser_run.add_argument(
        "--provider", required=True, help="Provider (groq, openai, ollama)"
    )
    parser_run.add_argument(
        "--workflow_type",
        required=True,
        choices=["cot", "tot", "fluid"],
        help="Type of workflow to run",
    )
    parser_run.add_argument(
        "--base_url_or_api_key",
        required=True,
        help="Base URL or API key for the provider",
    )
    parser_run.add_argument(
        "--model", required=True, help="Model to use in the workflow"
    )
    parser_run.add_argument("--options", help="Options for the workflow (JSON string)")
    parser_run.add_argument("--agents", help="Path to agents JSON file")
    parser_run.add_argument(
        "--task", required=True, help="Task description or path to task file"
    )

    args = parser.parse_args()

    if args.command == "serve":
        serve(args.host, args.port)
    elif args.command == "interface":
        from .__main__ import main as interface_main

        interface_main()
    elif args.command == "run":
        run_workflow(args)
    else:
        parser.print_help()


def serve(host, port):
    from .server import run_server

    run_server(host, port)


def run_workflow(args):
    from .runner import run_workflow_from_args

    run_workflow_from_args(args)


if __name__ == "__main__":
    main()
