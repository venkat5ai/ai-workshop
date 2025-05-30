import os
import yaml
import logging
import json
from functools import partial
import re

# LangChain Imports
from langchain.tools import Tool
from langchain_community.utilities.requests import RequestsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LC_Document

# New: For OpenAPI spec validation
from openapi_spec_validator import validate_spec
# Make OpenAPIValidationError robustly importable
OpenAPIValidationError = None
try:
    from openapi_spec_validator import OpenAPIValidationError
except ImportError:
    try:
        from openapi_spec_validator.validation import OpenAPIValidationError
    except ImportError:
        pass

logger = logging.getLogger(__name__)

if OpenAPIValidationError is None:
    logger.warning("Could not import 'OpenAPIValidationError' from 'openapi_spec_validator' or 'openapi_spec_validator.validation'. "
                   "OpenAPI validation errors will be caught by a generic Exception. Please ensure 'openapi-spec-validator' is correctly installed for your version.")
    OpenAPIValidationError = Exception


class ToolManager:
    """
    Manages the creation, storage, and retrieval of LangChain Tools from OpenAPI specifications.
    Tool descriptions are stored in a ChromaDB vector store for efficient retrieval.
    """
    def __init__(self, chroma_db_directory: str, embedding_function, tool_collection_name: str = "tool_descriptions_collection"):
        self.chroma_db_directory = chroma_db_directory
        self.embedding_function = embedding_function
        self.tool_collection_name = tool_collection_name
        self.tool_db = None
        self.all_available_tools: dict[str, Tool] = {}
        self.initialize_tool_db()

    def initialize_tool_db(self):
        """Initializes or loads the ChromaDB for tool descriptions."""
        try:
            os.makedirs(self.chroma_db_directory, exist_ok=True)
            
            if os.path.exists(self.chroma_db_directory) and os.listdir(self.chroma_db_directory):
                logger.info(f"Loading existing ChromaDB for tools from: {self.chroma_db_directory}")
                self.tool_db = Chroma(
                    persist_directory=self.chroma_db_directory,
                    embedding_function=self.embedding_function,
                    collection_name=self.tool_collection_name
                )
            else:
                logger.info(f"Creating new ChromaDB for tools at: {self.chroma_db_directory}")
                self.tool_db = Chroma(
                    embedding_function=self.embedding_function,
                    persist_directory=self.chroma_db_directory,
                    collection_name=self.tool_collection_name
                )
            logger.info(f"Tool ChromaDB initialized/loaded successfully from {self.chroma_db_directory}.")
        except Exception as e:
            logger.error(f"Failed to initialize Tool ChromaDB at {self.chroma_db_directory}: {e}", exc_info=True)
            self.tool_db = None

    def _create_langchain_tool_from_openapi_operation(
        self,
        base_url: str,
        path: str,
        method: str,
        operation_id: str,
        summary: str,
        parameters: list,
        requests_wrapper: RequestsWrapper,
        operation: dict
    ) -> Tool:
        """
        Internal helper to create a single LangChain Tool from an OpenAPI operation.
        """
        # CRITICAL FIX: Most precise tool name generation for Gemini compatibility
        # Allowed characters by Gemini: a-z, A-Z, 0-9, underscores (_), dots (.), dashes (-)
        # Must start with a letter or an underscore. Max 64 characters.

        raw_name_source = operation_id or f"{method.lower()}_{path.replace('/', '_').strip('_')}"
        logger.debug(f"Raw name source for tool: '{raw_name_source}'")

        # Step 1: Replace any character NOT explicitly allowed by Gemini's tool naming rules with an underscore.
        # This includes spaces, and other special symbols.
        # The regex now correctly allows '.', '-', '_'
        temp_name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', raw_name_source)
        logger.debug(f"After initial disallowed char replacement: '{temp_name}'")

        # Step 2: Normalize multiple consecutive separators (._-) into a single underscore.
        # This prevents names like "get..user--id" or "get___user"
        temp_name = re.sub(r'[._-]+', '_', temp_name)
        logger.debug(f"After normalizing separators: '{temp_name}'")

        # Step 3: Ensure it starts with a letter or an underscore. If not, prepend an underscore.
        if not re.match(r'^[a-zA-Z_]', temp_name):
            temp_name = '_' + temp_name
            logger.debug(f"After ensuring valid start char: '{temp_name}'")
        
        # Step 4: Remove any leading/trailing underscores that might remain
        # (e.g., if the original name was "_-tool-." this makes it "tool")
        # We only strip underscores as dots and dashes could be part of the valid name.
        temp_name = temp_name.strip('_') 
        logger.debug(f"After stripping leading/trailing underscores: '{temp_name}'")

        # Step 5: Convert to lowercase for consistency and simplicity.
        tool_name = temp_name.lower()
        logger.debug(f"After lowercase conversion: '{tool_name}'")

        # Step 6: Final check for empty name after extensive sanitization.
        if not tool_name:
             tool_name = f"tool_{abs(hash(raw_name_source)) % (10**6)}" # Fallback to a generic unique name
             logger.warning(f"Generated empty tool name from '{raw_name_source}'. Fallback to '{tool_name}'.")
             logger.debug(f"Fallback generated tool name: '{tool_name}'")

        # Step 7: Truncate to max 64 characters. Append a short hash if truncated to maintain uniqueness.
        if len(tool_name) > 64:
            unique_hash = str(abs(hash(raw_name_source)) % (10**6)) # Max 6 digits
            hash_len_with_sep = len(unique_hash) + 1 # +1 for the underscore separator
            
            truncated_len = 64 - hash_len_with_sep
            if truncated_len < 1: 
                truncated_len = 1 
            
            tool_name = tool_name[:truncated_len] + '_' + unique_hash
            # Final check to ensure it's still <= 64 after appending hash (should be)
            if len(tool_name) > 64:
                tool_name = tool_name[:64]
            logger.warning(f"Truncated tool name '{raw_name_source}' to '{tool_name}' due to 64-character limit and appended hash.")
        
        # Step 8: Final safety check for starting character after all operations
        # This addresses the edge case where truncation or hash appending might somehow lead to an invalid start.
        if not re.match(r'^[a-zA-Z_]', tool_name):
            tool_name = '_' + tool_name
            tool_name = tool_name[:64] # Re-truncate if prepending made it too long
            logger.warning(f"Final safety prepend: Tool name '{raw_name_source}' became '{tool_name}' after last check.")

        # CRITICAL ADDITION: Log the final tool_name being used here and the original name
        logger.error(f"DEBUGGING TOOL NAME ISSUE: Attempting to create tool with FINAL name: '{tool_name}' (Original source: '{raw_name_source}')")


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
                return f"Error processing LLM arguments: {e}. Input was: {json.dumps(args)}. Details: {e}"

            current_path_formatted = path
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
                        current_path_formatted = current_path_formatted.replace(f"{{{param_name}}}", str(param_value))
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


            full_url = f"{base_url}{current_path_formatted}"
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

    def load_openapi_spec_and_create_tools(
        self, spec_path: str, requests_wrapper: RequestsWrapper
    ) -> list[Tool]:
        """
        Loads an OpenAPI spec from a YAML file, validates it, creates LangChain Tools,
        and stores their descriptions in ChromaDB.
        Returns the created Tool objects (not necessarily all of them if filtered).
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
        except OpenAPIValidationError as e: # This is the robustly imported/aliased ValidationError
            logger.error(f"OpenAPI Spec Validation Error: {e.message} at {e.path}. Skipping tool creation.")
            return []
        except Exception as e: # Catch any other potential errors during validation
            logger.error(f"Unexpected error during OpenAPI spec validation: {e}. Skipping tool creation.", exc_info=True)
            return []


        base_url = "https://api.github.com"
        if "servers" in spec_dict and spec_dict["servers"]:
            base_url = spec_dict["servers"][0].get("url", base_url)
            if base_url.endswith('/') and any(p.startswith('/') for p in spec_dict.get("paths", {}).keys()):
                base_url = base_url.rstrip('/')
        logger.info(f"Inferred OpenAPI base URL: {base_url}")

        created_tools = []
        tool_documents = [] # For storing in ChromaDB

        # --- Filtering for GitHub API (can be refined or removed for tool retrieval) ---
        unsupported_path_keywords = [
            "/advisories", "/app-manifests/", "/app/hook/", "/app/installation-requests",
            "/app/installations/", "/applications/", "/credentials/", "/marketplace_listing/",
            "/enterprises/", "/orgs/{org}/actions/", "/orgs/{org}/attestations/",
            "/orgs/{org}/blocks/", "/orgs/{org}/campaigns/", "/orgs/{org}/code-scanning/",
            "/orgs/{org}/code-security/", "/orgs/{org}/codespaces/", "/orgs/{org}/copilot/",
            "/orgs/{org}/migrations/", "/orgs/{org}/oidc/", "/orgs/{org}/packages/",
            "/orgs/{org}/private-registries/", "/orgs/{org}/rulesets/",
            "/orgs/{org}/security-managers/", "/orgs/{org}/secret-scanning/",
            "/orgs/{org}/teams/", "/repos/{owner}/{repo}/actions/",
            "/repos/{owner}/{repo}/code-scanning/", "/repos/{owner}/{repo}/codespaces/",
            "/repos/{owner}/{repo}/dependency-graph/", "/repos/{owner}/{repo}/deployments/",
            "/repos/{owner}/{repo}/environments/", "/repos/{owner}/{repo}/forks",
            "/repos/{owner}/{repo}/git/", "/repos/{owner}/{repo}/hooks",
            "/repos/{owner}/{repo}/invitations", "/repos/{owner}/{repo}/keys",
            "/repos/{owner}/{repo}/labels", "/repos/{owner}/{repo}/lfs",
            "/repos/{owner}/{repo}/merge-queue", "/repos/{owner}/{repo}/milestones",
            "/repos/{owner}/{repo}/pages", "/repos/{owner}/{repo}/private-vulnerability-reporting",
            "/repos/{owner}/{repo}/secret-scanning/", "/repos/{owner}/{repo}/secrets/",
            "/repos/{owner}/{repo}/tags", "/repos/{owner}/{repo}/tarball/",
            "/repos/{owner}/{repo}/topics", "/repos/{owner}/{repo}/traffic",
            "/repos/{owner}/{repo}/transfer", "/repos/{owner}/{repo}/vulnerability-alerts",
            "/repos/{owner}/{repo}/zipball/", "/scim/", "/user/emails", "/user/followers",
            "/user/following", "/user/keys", "/user/migrations", "/user/orgs",
            "/user/repository_invitations", "/user/starred", "/user/subscriptions",
            "/user/teams", "/user/codespaces", "/user/installations",
            "/users/{username}/followers", "/users/{username}/following", "/users/{username}/gists",
            "/users/{username}/keys", "/users/{username}/starred", "/users/{username}/subscriptions",
            "/users/{username}/gpg_keys", "/search/"
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
                            tool = self._create_langchain_tool_from_openapi_operation( # Use self._create...
                                base_url, path, method, operation_id, summary, parameters, requests_wrapper, operation
                            )
                            created_tools.append(tool)
                            logger.info(f"Created tool: {tool.name} for {method.upper()} {path}")
                            
                            # Store tool description in ChromaDB
                            tool_doc = LC_Document(
                                page_content=tool.description,
                                metadata={
                                    "tool_name": tool.name,
                                    "operation_id": operation_id,
                                    "api_name": spec_path.split(os.sep)[-1].replace(".yml", ""),
                                    "source": spec_path
                                }
                            )
                            tool_documents.append(tool_doc)
                            self.all_available_tools[tool.name] = tool # Store tool object by name
                        except Exception as e:
                            logger.warning(f"Failed to create tool for {operation_id} ({method.upper()} {path}): {e}")
                    else:
                        logger.warning(f"Skipping operation without 'operationId': {method.upper()} {path}")
        
        if tool_documents and self.tool_db:
            try:
                self.tool_db.add_documents(tool_documents)
                logger.info(f"Added {len(tool_documents)} tool descriptions to ChromaDB.")
            except Exception as e:
                logger.error(f"Failed to add tool descriptions to ChromaDB: {e}", exc_info=True)
        elif not self.tool_db:
            logger.warning("Tool ChromaDB not initialized. Skipping adding tool descriptions to DB.")

        if not created_tools:
            logger.warning(f"No tools were created from the OpenAPI spec at {spec_path}.")
        return created_tools

    def get_relevant_tools(self, query: str, k: int = 5) -> list[Tool]:
        """
        Retrieves the top-k most relevant tools based on a semantic search against
        their descriptions in ChromaDB.
        """
        if not self.tool_db:
            logger.error("Tool ChromaDB is not initialized. Cannot retrieve tools.")
            return []
        if not query.strip():
            logger.warning("Empty query for tool retrieval. Returning no tools.")
            return []

        try:
            # Perform a similarity search on tool descriptions
            retrieved_docs = self.tool_db.similarity_search(query, k=k)
            
            relevant_tools = []
            for doc in retrieved_docs:
                tool_name = doc.metadata.get("tool_name")
                if tool_name and tool_name in self.all_available_tools:
                    relevant_tools.append(self.all_available_tools[tool_name])
                    logger.debug(f"Retrieved relevant tool: {tool_name}")
                else:
                    logger.warning(f"Retrieved document for tool '{tool_name}' but tool object not found in all_available_tools or metadata is missing.")
            
            logger.info(f"Retrieved {len(relevant_tools)} relevant tools for query: '{query}'")
            return relevant_tools
        except Exception as e:
            logger.error(f"Error during tool retrieval from ChromaDB for query '{query}': {e}", exc_info=True)
            return []