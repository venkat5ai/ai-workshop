import os
import yaml
import logging

logger = logging.getLogger(__name__)

import json
from functools import partial

# LangChain Imports
from langchain.tools import Tool
from langchain_community.utilities.requests import RequestsWrapper # For making HTTP requests

# New: For OpenAPI spec validation
from openapi_spec_validator import validate_spec 

# CRITICAL FIX: Make OpenAPIValidationError import robust
OpenAPIValidationError = None # Initialize to None
try:
    from openapi_spec_validator import OpenAPIValidationError # Try direct import first (more common now)
except ImportError:
    try:
        from openapi_spec_validator import OpenAPIValidationError
    except ImportError:
        logger.warning("Could not import 'OpenAPIValidationError' from 'openapi_spec_validator' or 'openapi_spec_validator.validation'. "
                       "OpenAPI validation errors will be caught by a generic Exception. Please ensure 'openapi-spec-validator' is correctly installed for your version.")
        # If it cannot be imported, we'll fall back to catching a generic Exception for validation errors
        OpenAPIValidationError = Exception 

def create_langchain_tool_from_openapi_operation(
    base_url: str,
    path: str,
    method: str,
    operation_id: str,
    summary: str,
    parameters: list,
    requests_wrapper: RequestsWrapper,
    operation: dict # Pass the full operation dict for requestBody handling
) -> Tool:
    """
    Creates a single LangChain Tool from a specific OpenAPI operation.
    """
    tool_name = operation_id.replace('-', '_').replace('.', '_').lower()

    param_descriptions = []
    for param in parameters:
        param_name = param.get('name')
        param_in = param.get('in')
        param_required = param.get('required', False)
        param_desc = param.get('description', '')
        param_type = param.get('schema', {}).get('type')
        if not param_type and '$ref' in param.get('schema', {}):
            param_type = param['schema']['$ref'].split('/')[-1]
            param_desc = f"Complex type: {param_type}. " + param_desc

        desc_part = f"`{param_name}` ({param_type}, in {param_in})"
        if param_required:
            desc_part += " [REQUIRED]"
        if param_desc:
            desc_part += f": {param_desc}"
        if param.get('schema', {}).get('enum'):
            desc_part += f" (Allowed values: {', '.join(map(str, param['schema']['enum']))})"
        param_descriptions.append(desc_part)

    # Handle requestBody if present in the operation
    request_body_description = ""
    if 'requestBody' in operation:
        req_body = operation['requestBody']
        req_body_required = req_body.get('required', False)
        req_body_desc = req_body.get('description', '')
        
        content_types = list(req_body.get('content', {}).keys())
        if content_types:
            content_type = content_types[0]
            schema_ref = req_body['content'][content_type].get('schema', {}).get('$ref')
            if schema_ref:
                schema_name = schema_ref.split('/')[-1]
                request_body_description = (
                    f"This operation requires a request body of type '{content_type}' "
                    f"with schema '{schema_name}'. "
                    f"Example input for request body: {{\"key\": \"value\"}} "
                    f"(refer to '{schema_name}' for exact structure)."
                )
            else:
                request_body_description = (
                    f"This operation requires a request body of type '{content_type}'. "
                    f"Example input for request body: {{\"key\": \"value\"}}."
                )
        if req_body_required:
            request_body_description = "[REQUIRED] " + request_body_description
        if req_body_desc:
            request_body_description = f"{req_body_desc}. " + request_body_description
        
        if request_body_description:
            param_descriptions.append(f"Request Body: {request_body_description}")


    tool_description = (
        f"{summary}\n\n"
        f"Input: A JSON string containing the parameters for this operation.\n"
        f"Parameters:\n" + "\n".join([f"  - {d}" for d in param_descriptions]) + "\n"
        f"Example JSON input: {{\"param_name\": \"value\", ...}}"
    )
    
    if parameters and not request_body_description:
        example_params = {p['name']: f"<{p['schema'].get('type', 'value')}>" for p in parameters if p['in'] in ['path', 'query']}
        tool_description += f"\nSpecific example: {json.dumps(example_params)}"


    def _dynamic_api_call(json_args_str: str, method_type: str) -> str:
        # print(f"\n--- DEBUG: Inside _dynamic_api_call for {tool_name} ({method_type.upper()}) ---")
        # print(f"--- DEBUG: Received JSON args: {json_args_str} ---")

        try:
            llm_args = json.loads(json_args_str)
            if '__arg1' in llm_args: 
                args = json.loads(llm_args['__arg1'])
            else:
                args = llm_args
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError for tool '{tool_name}': {e}. Input: {json_args_str}")
            return f"Error: Invalid JSON input. Please provide parameters as a valid JSON string. Details: {e}"
        except Exception as e:
            logger.error(f"Error processing LLM args for tool '{tool_name}': {e}. Input: {json_args_str}")
            return f"Error processing LLM arguments: {e}. Input was: {json_args_str}"

        current_path = path
        query_params = {}
        request_body_payload = None 
        is_json_body = False

        for param in parameters:
            param_name = param['name']
            param_in = param['in']
            param_required = param.get('required', False)

            param_value = args.get(param_name)

            if param_required and param_value is None:
                return f"Error: Required parameter '{param_name}' is missing for {tool_name}."

            if param_value is not None:
                if param_in == 'path':
                    current_path = current_path.replace(f"{{{param_name}}}", str(param_value))
                elif param_in == 'query':
                    query_params[param_name] = str(param_value)

        if 'requestBody' in operation:
            req_body_content = operation['requestBody'].get('content', {})
            if 'application/json' in req_body_content:
                is_json_body = True
                request_body_payload = args 
            elif 'application/x-www-form-urlencoded' in req_body_content or 'multipart/form-data' in req_body_content:
                request_body_payload = args
            else:
                logger.warning(f"Unsupported requestBody content type for {tool_name}: {list(req_body_content.keys())}")
                return f"Error: Unsupported request body content type for {tool_name}."


        full_url = f"{base_url}{current_path}"
        logger.info(f"Making API call: {method_type.upper()} {full_url} with query {query_params} and body payload (if any).")

        try:
            requester_func = getattr(requests_wrapper, method_type.lower())

            if method_type.lower() in ['get', 'delete']:
                response_text = requester_func(
                    url=full_url,
                    params=query_params,
                )
            elif method_type.lower() in ['post', 'put', 'patch']:
                if is_json_body:
                    response_text = requester_func(
                        url=full_url,
                        params=query_params, 
                        json=request_body_payload, 
                    )
                else:
                    response_text = requester_func(
                        url=full_url,
                        params=query_params,
                        data=request_body_payload, 
                    )
            else:
                return f"Error: Unsupported HTTP method '{method_type}' for {tool_name}."

            if not response_text.strip():
                logger.warning(f"Empty response text from API for {tool_name}.")
                return "Error: Empty response from API."
            
            try:
                parsed_response = json.loads(response_text)
                if isinstance(parsed_response, dict) and ('message' in parsed_response or 'error' in parsed_response):
                     logger.warning(f"API returned a JSON error for {tool_name}: {response_text[:200]}")
                     return f"API error: {parsed_response.get('message', 'Unknown error message')}. Raw response: {response_text[:200]}"
                
                logger.info(f"API call for {tool_name} successful, returning JSON.")
                return json.dumps(parsed_response, indent=2) 

            except json.JSONDecodeError:
                logger.info(f"API response for {tool_name} is not JSON, returning raw text.")
                return response_text 

        except Exception as e:
            logger.error(f"Error calling API for {tool_name} ({summary}): {e}", exc_info=True)
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
    Includes basic validation and filtering.
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

    # --- Validate OpenAPI Spec ---
    try:
        validate_spec(spec_dict)
        logger.info("OpenAPI spec validated successfully (v3.x).")
    # CRITICAL FIX: Catch OpenAPIValidationError specifically if it's found, else catch generic Exception
    except OpenAPIValidationError as e: 
        logger.error(f"OpenAPI Spec Validation Error: {e.message} at {e.path}. Skipping tool creation.")
        return []
    except Exception as e: # Catch any other potential errors during validation
        logger.error(f"Unexpected error during OpenAPI spec validation: {e}. Skipping tool creation.", exc_info=True)
        return []


    base_url = "https://api.github.com" # Default to GitHub API base URL
    if "servers" in spec_dict and spec_dict["servers"]:
        base_url = spec_dict["servers"][0].get("url", base_url)
        if base_url.endswith('/') and any(p.startswith('/') for p in spec_dict.get("paths", {}).keys()):
            base_url = base_url.rstrip('/')
    logger.info(f"Inferred OpenAPI base URL: {base_url}")

    tools = []
    
    # --- Filtering for GitHub API to manage LLM context size for Phase 1 ---
    unsupported_path_keywords = [
        "/advisories", # Complex parameters
        "/app-manifests/", 
        "/app/hook/", 
        "/app/installation-requests",
        "/app/installations/",
        "/applications/", 
        "/credentials/", 
        "/marketplace_listing/", 
        "/enterprises/", 
        "/orgs/{org}/actions/",
        "/orgs/{org}/attestations/",
        "/orgs/{org}/blocks/",
        "/orgs/{org}/campaigns/",
        "/orgs/{org}/code-scanning/",
        "/orgs/{org}/code-security/",
        "/orgs/{org}/codespaces/",
        "/orgs/{org}/copilot/",
        "/orgs/{org}/migrations/",
        "/orgs/{org}/oidc/",
        "/orgs/{org}/packages/",
        "/orgs/{org}/private-registries/",
        "/orgs/{org}/rulesets/",
        "/orgs/{org}/security-managers/",
        "/orgs/{org}/secret-scanning/",
        "/orgs/{org}/teams/",
        "/repos/{owner}/{repo}/actions/",
        "/repos/{owner}/{repo}/code-scanning/",
        "/repos/{owner}/{repo}/codespaces/",
        "/repos/{owner}/{repo}/dependency-graph/",
        "/repos/{owner}/{repo}/deployments/",
        "/repos/{owner}/{repo}/environments/",
        "/repos/{owner}/{repo}/forks", 
        "/repos/{owner}/{repo}/git/",
        "/repos/{owner}/{repo}/hooks", 
        "/repos/{owner}/{repo}/invitations",
        "/repos/{owner}/{repo}/keys",
        "/repos/{owner}/{repo}/labels",
        "/repos/{owner}/{repo}/lfs",
        "/repos/{owner}/{repo}/merge-queue",
        "/repos/{owner}/{repo}/milestones",
        "/repos/{owner}/{repo}/pages",
        "/repos/{owner}/{repo}/private-vulnerability-reporting",
        "/repos/{owner}/{repo}/secret-scanning/",
        "/repos/{owner}/{repo}/secrets/",
        "/repos/{owner}/{repo}/tags",
        "/repos/{owner}/{repo}/tarball/",
        "/repos/{owner}/{repo}/topics",
        "/repos/{owner}/{repo}/traffic",
        "/repos/{owner}/{repo}/transfer",
        "/repos/{owner}/{repo}/vulnerability-alerts",
        "/repos/{owner}/{repo}/zipball/",
        "/scim/",
        "/user/emails", 
        "/user/followers", 
        "/user/following", 
        "/user/keys", 
        "/user/migrations", 
        "/user/orgs", 
        "/user/repository_invitations", 
        "/user/starred", 
        "/user/subscriptions", 
        "/user/teams", 
        "/user/codespaces", 
        "/user/installations", 
        "/users/{username}/followers", 
        "/users/{username}/following", 
        "/users/{username}/gists", 
        "/users/{username}/keys", 
        "/users/{username}/starred", 
        "/users/{username}/subscriptions", 
        "/users/{username}/gpg_keys",
        "/search/", 
    ]


    for path, path_item in spec_dict.get("paths", {}).items():
        if any(path.startswith(kw) for kw in unsupported_path_keywords):
            logger.info(f"Skipping heavily filtered or sensitive path: {path}")
            continue

        for method, operation in path_item.items():
            if method.lower() in ["get", "post", "put", "delete", "patch"]:
                operation_id = operation.get("operationId")
                summary = operation.get("summary") or f"{method.upper()} {path}"
                parameters = operation.get("parameters", [])

                if operation_id:
                    if any('$ref' in p.get('schema', {}) and p['schema']['$ref'].split('/')[-1] not in ['string', 'integer', 'boolean'] for p in parameters):
                        logger.info(f"Skipping operation {operation_id} due to complex $ref in parameters: {path}")
                        continue
                    
                    try:
                        tool = create_langchain_tool_from_openapi_operation(
                            base_url,
                            path,
                            method,
                            operation_id,
                            summary,
                            parameters,
                            requests_wrapper,
                            operation
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