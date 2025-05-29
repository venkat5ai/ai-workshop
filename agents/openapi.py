import os
import yaml
import logging
import sys
import json
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities.requests import RequestsWrapper # For making HTTP requests

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # NEW: For GitHub API authentication
GEMINI_MODEL_TO_USE = "models/gemini-1.5-flash"
CONFIG_DIR = "./config" # Directory where OpenAPI spec files are stored
GITHUB_SPEC_FILENAME = "api.github.com.yml" # Your full GitHub spec

# --- Helper function to dynamically create LangChain Tools from an OpenAPI operation ---
def create_langchain_tool_from_openapi_operation(
    base_url: str,
    path: str,
    method: str,
    operation_id: str,
    summary: str,
    parameters: list,
    requests_wrapper: RequestsWrapper
) -> Tool:
    """
    Creates a single LangChain Tool from a specific OpenAPI operation.

    Args:
        base_url (str): The base URL for the API.
        path (str): The API path (e.g., "/users/{username}/repos").
        method (str): The HTTP method (e.g., "get", "post").
        operation_id (str): The unique operation ID from the OpenAPI spec.
        summary (str): A summary of the operation for the tool description.
        parameters (list): A list of parameter definitions from the OpenAPI spec.
        requests_wrapper (RequestsWrapper): An initialized RequestsWrapper instance.

    Returns:
        Tool: A LangChain Tool instance.
    """
    tool_name = operation_id.replace('-', '_').replace('.', '_').lower()

    param_descriptions = []
    for param in parameters:
        param_name = param.get('name')
        param_in = param.get('in')
        param_required = param.get('required', False)
        param_desc = param.get('description', '')
        param_type = param.get('schema', {}).get('type') # Use 'type' directly
        # Check for $ref in schema, but for complex schemas, LLM might still struggle without full resolution
        if not param_type and '$ref' in param.get('schema', {}):
            param_type = param['schema']['$ref'].split('/')[-1] # Extract name from $ref
            param_desc = f"Complex type: {param_type}. " + param_desc


        desc_part = f"`{param_name}` ({param_type}, in {param_in})"
        if param_required:
            desc_part += " [REQUIRED]"
        if param_desc:
            desc_part += f": {param_desc}"
        if param.get('schema', {}).get('enum'):
            desc_part += f" (Allowed values: {', '.join(map(str, param['schema']['enum']))})"
        param_descriptions.append(desc_part)

    tool_description = (
        f"{summary}\n\n"
        f"Input: A JSON string containing the parameters for this operation.\n"
        f"Parameters:\n" + "\n".join([f"  - {d}" for d in param_descriptions]) + "\n"
        f"Example JSON input: {{\"param_name\": \"value\", ...}}"
    )

    def _dynamic_api_call(json_args_str: str, method_type: str) -> str:
        print(f"\n--- DEBUG: Inside _dynamic_api_call for {tool_name} ({method_type.upper()}) ---")
        print(f"--- DEBUG: Received JSON args: {json_args_str} ---")

        try:
            llm_args = json.loads(json_args_str)
            # Handle the '__arg1' wrapper that Gemini sometimes adds for single arguments
            if '__arg1' in llm_args:
                args = json.loads(llm_args['__arg1'])
            else:
                args = llm_args
        except json.JSONDecodeError as e:
            print(f"--- DEBUG: JSONDecodeError: {e} ---")
            return f"Error: Invalid JSON input. Please provide parameters as a valid JSON string. Details: {e}"
        except Exception as e:
            print(f"--- DEBUG: Error processing LLM args: {e} ---")
            return f"Error processing LLM arguments: {e}. Input was: {json_args_str}"

        # Populate path and query parameters
        current_path = path
        query_params = {}
        request_body = {}
        # Note: headers are managed by RequestsWrapper constructor for auth
        
        for param in parameters:
            param_name = param['name']
            param_in = param['in']
            param_required = param.get('required', False)

            param_value = args.get(param_name)

            if param_required and param_value is None:
                # LLM should handle this via prompt, but we can catch it here too.
                return f"Error: Required parameter '{param_name}' is missing for {tool_name}."

            if param_value is not None:
                if param_in == 'path':
                    current_path = current_path.replace(f"{{{param_name}}}", str(param_value))
                elif param_in == 'query':
                    query_params[param_name] = str(param_value)
                elif param_in == 'header':
                    # Headers from spec are usually fixed, dynamic ones passed here might conflict
                    # For now, let RequestsWrapper handle fixed auth headers.
                    pass 
                elif 'requestBody' in operation and 'content' in operation['requestBody']:
                    # This is a simplified way to handle request bodies.
                    # A more robust solution would inspect content type and schema.
                    # For complex body types, LLM needs to provide a structured JSON.
                    request_body = args # Assume entire args is the request body for simplicity
                elif param_in == 'body': # Older OpenAPI versions
                    request_body = args


        full_url = f"{base_url}{current_path}"
        logger.info(f"Making API call: {method_type.upper()} {full_url} with query {query_params} and body {request_body}")

        try:
            requester_func = getattr(requests_wrapper, method_type.lower())

            if method_type.lower() in ['get', 'delete']:
                response_text = requester_func(
                    url=full_url,
                    params=query_params,
                )
            elif method_type.lower() in ['post', 'put', 'patch']:
                # The LLM's 'args' might contain parameters for the body or form data
                # If request_body is a dict, send as json, otherwise as data
                if request_body and isinstance(request_body, dict):
                    response_text = requester_func(
                        url=full_url,
                        params=query_params, # Query params still apply
                        json=request_body,   # Send as JSON body
                    )
                else: # Fallback for other cases or if request_body is a string
                    response_text = requester_func(
                        url=full_url,
                        params=query_params,
                        data=request_body, # Send as data (e.g., form-urlencoded)
                    )
            else:
                return f"Error: Unsupported HTTP method '{method_type}' for {tool_name}."

            # Check if response_text is empty or indicates an error
            if not response_text.strip():
                print("--- DEBUG: Empty response text from API. ---")
                return "Error: Empty response from API."
            
            # Try to parse the response text as JSON to check for API errors or valid data
            try:
                parsed_response = json.loads(response_text)
                if isinstance(parsed_response, dict) and ('message' in parsed_response or 'error' in parsed_response):
                     print(f"--- DEBUG: API returned a JSON error: {response_text[:200]} ---")
                     # GitHub often uses 'message' for errors
                     return f"API error: {parsed_response.get('message', 'Unknown error message')}. Raw response: {response_text[:200]}"
                
                print(f"--- DEBUG: API call successful, returning JSON. ---")
                return json.dumps(parsed_response, indent=2) # Pretty print for readability for LLM

            except json.JSONDecodeError:
                # If it's not JSON, it might be raw text (e.g., gitignore template, Octocat ASCII)
                print(f"--- DEBUG: API response is not JSON, returning raw text. ---")
                return response_text # Return raw text if not JSON

        except Exception as e:
            logger.error(f"Error calling API for {tool_name} ({summary}): {e}", exc_info=True)
            print(f"--- DEBUG: API call failed: {e} ---")
            return f"Error executing API call '{tool_name}': {e}. Input was: {json.dumps(args)}. Details: {e}"

    return Tool(
        name=tool_name,
        func=partial(_dynamic_api_call, method_type=method),
        description=tool_description
    )

def load_and_create_tools_from_openapi_spec(
    spec_path: str, requests_wrapper: RequestsWrapper
) -> list[Tool]:
    """
    Loads an OpenAPI spec from a YAML file and creates LangChain Tools for its operations.
    """
    logger.info(f"Loading and parsing OpenAPI spec from {spec_path}...")
    spec_dict = {}
    try:
        with open(spec_path, "r") as f:
            spec_dict = yaml.safe_load(f)
        logger.info("OpenAPI spec loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: OpenAPI spec file not found at {spec_path}.")
        return []
    except yaml.YAMLError as e:
        logger.error(f"Error: Failed to parse YAML from {spec_path}: {e}. Ensure it's valid YAML.", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Failed to load or parse OpenAPI spec from {spec_path}: {e}", exc_info=True)
        return []

    # Parse the spec to extract base URL
    base_url = "http://localhost" # Default fallback
    if "servers" in spec_dict and spec_dict["servers"]:
        base_url = spec_dict["servers"][0].get("url", base_url)
        if base_url.endswith('/') and any(p.startswith('/') for p in spec_dict.get("paths", {}).keys()):
            base_url = base_url.rstrip('/')

    tools = []
    # Filter out operations that resolve external schemas or complex types for now
    # This is a heuristic to reduce complexity for the LLM
    unsupported_path_keywords = [
        "/advisories", # Complex parameters
        "/app-manifests", # POST, requires code
        "/app/hook", # Webhook related, usually not for direct queries
        "/app/installation-requests",
        "/app/installations",
        "/app/installations/",
        "/applications/", # OAuth token management
        "/assignments", # Classroom specific, needs classroom ID
        "/classrooms", # Classroom specific, needs classroom ID
        "/credentials", # Revoke credentials - sensitive
        "/marketplace_listing", # Marketplace specific
        "/enterprises/", # Enterprise specific, needs enterprise ID
        "/orgs/{org}/actions/", # Many complex actions, needs org and often other specific IDs
        "/orgs/{org}/attestations/", # Complex subject_digest
        "/orgs/{org}/blocks/", # User blocking
        "/orgs/{org}/campaigns/", # Campaign management
        "/orgs/{org}/code-scanning/", # Code scanning specific
        "/orgs/{org}/code-security/", # Code security specific
        "/orgs/{org}/codespaces/", # Codespaces specific
        "/orgs/{org}/copilot/", # Copilot specific
        "/orgs/{org}/events", # Events related
        "/orgs/{org}/migrations/", # Migrations related
        "/orgs/{org}/oidc/", # OIDC related
        "/orgs/{org}/packages/", # Packages related
        "/orgs/{org}/private-registries/", # Private registries
        "/orgs/{org}/rulesets/", # Rulesets
        "/orgs/{org}/security-managers/", # Security managers
        "/orgs/{org}/secret-scanning/", # Secret scanning
        "/orgs/{org}/teams/", # Teams related
        "/repos/{owner}/{repo}/actions/", # Repo actions
        "/repos/{owner}/{repo}/code-scanning/", # Repo code scanning
        "/repos/{owner}/{repo}/codespaces/", # Repo codespaces
        "/repos/{owner}/{repo}/community/profile", # Complex response
        "/repos/{owner}/{repo}/dependency-graph/", # Dependency graph
        "/repos/{owner}/{repo}/deployments/", # Deployments
        "/repos/{owner}/{repo}/environments/", # Environments
        "/repos/{owner}/{repo}/forks", # Forking
        "/repos/{owner}/{repo}/git/", # Raw git
        "/repos/{owner}/{repo}/hooks", # Webhooks
        "/repos/{owner}/{repo}/invitations", # Invitations
        "/repos/{owner}/{repo}/keys", # Deploy keys
        "/repos/{owner}/{repo}/labels", # Labels
        "/repos/{owner}/{repo}/lfs", # LFS
        "/repos/{owner}/{repo}/merge-queue", # Merge queue
        "/repos/{owner}/{repo}/milestones", # Milestones
        "/repos/{owner}/{repo}/pages", # Pages
        "/repos/{owner}/{repo}/private-vulnerability-reporting", # Private vuln reporting
        "/repos/{owner}/{repo}/pulls/", # Pull requests - can be complex
        "/repos/{owner}/{repo}/secret-scanning/", # Secret scanning
        "/repos/{owner}/{repo}/secrets/", # Repo secrets
        "/repos/{owner}/{repo}/tags", # Tags
        "/repos/{owner}/{repo}/tarball/", # Tarball
        "/repos/{owner}/{repo}/topics", # Topics
        "/repos/{owner}/{repo}/traffic", # Traffic
        "/repos/{owner}/{repo}/transfer", # Transfer repo
        "/repos/{owner}/{repo}/vulnerability-alerts", # Vulnerability alerts
        "/repos/{owner}/{repo}/zipball/", # Zipball
        "/scim/", # SCIM
        "/search/code", # Complex search parameters
        "/user/emails", # User emails - sensitive
        "/user/followers", # User followers
        "/user/following", # User following
        "/user/keys", # User keys
        "/user/migrations", # User migrations
        "/user/orgs", # User orgs
        "/user/repository_invitations", # User repo invitations
        "/user/starred", # User starred repos
        "/user/subscriptions", # User subscriptions
        "/user/teams", # User teams
        "/user/codespaces", # User codespaces
        "/user/installations", # User installations
        "/users/{username}/followers", # User followers
        "/users/{username}/following", # User following
        "/users/{username}/gists", # User gists
        "/users/{username}/keys", # User keys
        "/users/{username}/starred", # User starred
        "/users/{username}/subscriptions", # User subscriptions
        "/users/{username}/gpg_keys", # User GPG keys
    ]


    for path, path_item in spec_dict.get("paths", {}).items():
        # Skip paths that are known to be problematic or require complex auth/setup
        if any(path.startswith(kw) for kw in unsupported_path_keywords):
            logger.info(f"Skipping potentially complex/sensitive path: {path}")
            continue

        for method, operation in path_item.items():
            if method.lower() in ["get", "post", "put", "delete", "patch"]:
                operation_id = operation.get("operationId")
                summary = operation.get("summary") or f"{method.upper()} {path}"
                parameters = operation.get("parameters", [])

                if operation_id:
                    # Filter out operations that have $ref in parameters that aren't simple strings/integers
                    # This is a heuristic, not a full schema resolver.
                    if any('$ref' in p.get('schema', {}) and p['schema']['$ref'].split('/')[-1] not in ['string', 'integer', 'boolean'] for p in parameters):
                        logger.info(f"Skipping operation {operation_id} due to complex $ref in parameters: {path}")
                        continue
                    
                    # Also skip if requestBody exists for simplicity, as it adds complexity
                    if 'requestBody' in operation:
                        logger.info(f"Skipping operation {operation_id} due to requestBody: {path}")
                        continue

                    try:
                        tool = create_langchain_tool_from_openapi_operation(
                            base_url,
                            path,
                            method,
                            operation_id,
                            summary,
                            parameters,
                            requests_wrapper
                        )
                        tools.append(tool)
                        logger.info(f"Created tool: {tool.name} for {method.upper()} {path}")
                    except Exception as e:
                        logger.warning(f"Failed to create tool for {operation_id} ({method.upper()} {path}): {e}")
                else:
                    logger.warning(f"Skipping operation without 'operationId': {method.upper()} {path}")
    
    if not tools:
        logger.warning(f"No tools were created from the OpenAPI spec at {spec_path}.")
    return tools

def main():
    if not GOOGLE_API_KEY:
        logger.error("Error: GOOGLE_API_KEY environment variable not set. Please set it.")
        sys.exit(1)
    
    # GITHUB_TOKEN is optional, but highly recommended for better rate limits and access
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        logger.info("GitHub API requests will be authenticated.")
    else:
        logger.warning("GITHUB_TOKEN environment variable not set. GitHub API requests may be rate-limited or fail for private resources.")


    # 1. Configure LLM (Gemini)
    logger.info("Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_TO_USE,
        temperature=0.0, # Keep temperature low for deterministic tool calling
        google_api_key=GOOGLE_API_KEY
    )
    logger.info("Gemini LLM initialized.")

    # 2. Create requests wrapper (reusable for all tools, with optional auth headers)
    logger.info("Creating RequestsWrapper...")
    requests_wrapper = RequestsWrapper(headers=headers)
    logger.info("RequestsWrapper created.")

    # 3. Load OpenAPI spec and create tools for GitHub API
    github_spec_path = os.path.join(CONFIG_DIR, GITHUB_SPEC_FILENAME)
    logger.info(f"Attempting to load GitHub API tools from: {github_spec_path}")
    github_tools = load_and_create_tools_from_openapi_spec(github_spec_path, requests_wrapper)

    if not github_tools:
        logger.error("No GitHub tools were loaded. Exiting.")
        sys.exit(1)

    all_tools = github_tools
    logger.info(f"Total tools available: {len(all_tools)}")

    # 4. Create AgentExecutor
    logger.info("Creating AgentExecutor...")
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant specialized in interacting with the GitHub API.
            You have access to various tools to query GitHub information.
            Use the provided tools to answer questions about GitHub.
            Always use the most appropriate tool based on the user's request.
            If a required parameter for a tool is missing, ask the user for it clearly.
            After calling a tool, summarize the relevant information from its output for the user.
            Do NOT try to answer based on your internal knowledge if a tool can provide a more accurate or up-to-date answer.
            If you cannot fulfill a request with the available tools, inform the user clearly.
            """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, all_tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        logger.info("AgentExecutor created.")
    except Exception as e:
        logger.error(f"Failed to create AgentExecutor: {e}", exc_info=True)
        sys.exit(1)

    # --- Interactive Chat Loop ---
    print("\n--- GitHub API Agent Interactive Chat ---")
    print("Ask me about GitHub (e.g., 'List public repositories for user octocat'). Type 'exit' to quit.")

    chat_history = []
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            logger.info(f"Invoking agent with query: '{user_input}'")
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})

            print("\n--- Agent Chain Details (Verbose) ---")
            if result.get('intermediate_steps'):
                for step in result['intermediate_steps']:
                    print(f"Thought: {getattr(step[0], 'log', 'N/A')}")
                    print(f"Tool: {getattr(step[0], 'tool', 'N/A')}")
                    print(f"Tool Input: {getattr(step[0], 'tool_input', 'N/A')}")
                    print(f"Observation: {step[1]}")
                    print("-" * 30)
            else:
                print("No intermediate steps returned. This might indicate the LLM did not call a tool or directly provided an answer.")

            final_response = result.get('output', 'No output found.')
            print("\nAgent: " + final_response)

            # Update chat history for context
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_response))

        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            print(f"Agent: An error occurred during execution: {e}. Please check logs for details.")

if __name__ == "__main__":
    main()