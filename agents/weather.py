import os
import yaml
import logging
import sys
import json 

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain Imports for Gemini and core agent building blocks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities.requests import RequestsWrapper 
from langchain.tools import Tool 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_TO_USE = "models/gemini-1.5-flash" 
WEATHER_SPEC_PATH = "./data/weather.yml" 

# Helper function to dynamically create LangChain Tool for the forecast endpoint
def create_weather_forecast_tool_from_dict(openapi_spec_dict: dict, requests_wrapper: RequestsWrapper) -> Tool:
    """
    Manually creates a LangChain Tool for the Open-Meteo weather forecast endpoint.
    Assumes there's a /v1/forecast GET endpoint in the spec.
    """
    base_url = "https://api.open-meteo.com" 
    if "servers" in openapi_spec_dict and openapi_spec_dict["servers"]:
        base_url = openapi_spec_dict["servers"][0].get("url", base_url)
        if base_url.endswith('/') and any(p.startswith('/') for p in openapi_spec_dict.get("paths", {}).keys()):
            base_url = base_url.rstrip('/')

    for path, path_item in openapi_spec_dict.get("paths", {}).items():
        if path == "/v1/forecast" and "get" in path_item:
            operation = path_item["get"]
            operation_id = operation.get("operationId", "get_v1_forecast")
            summary = operation.get("summary", "Get 7-day weather forecast")
            
            tool_name = operation_id.replace('-', '_').replace('.', '_').lower() 

            def _dynamic_api_call(json_args_str: str) -> str:
                # IMPORTANT: Add a print statement here to see if this function is ever called
                print(f"\n--- DEBUG: Inside _dynamic_api_call for {tool_name} ---")
                print(f"--- DEBUG: Received JSON args: {json_args_str} ---")

                try:
                    args = json.loads(json_args_str)
                except json.JSONDecodeError:
                    print("--- DEBUG: JSONDecodeError ---")
                    return f"Error: Invalid JSON input. Please provide parameters as a valid JSON string. Example: '{{\"latitude\": 37.77, \"longitude\": -122.41, \"current_weather\": true}}'"

                query_params = args
                full_url = f"{base_url}{path}" 

                logger.info(f"Making Weather API call: GET {full_url} with query {query_params}")
                
                try:
                    response = requests_wrapper.request(
                        method="GET",
                        url=full_url,
                        params=query_params, 
                    )
                    response.raise_for_status() 
                    print(f"--- DEBUG: API call successful, returning response text. Status: {response.status_code} ---")
                    return response.text 
                except Exception as e:
                    logger.error(f"Error calling Weather API for {tool_name} ({summary}): {e}", exc_info=True)
                    print(f"--- DEBUG: API call failed: {e} ---")
                    return f"Error executing API call '{tool_name}': {e}. Input was: {json_args_str}"
            
            return Tool(
                name=tool_name,
                func=_dynamic_api_call, 
                description=(
                    f"**This tool provides weather forecasts.** "
                    f"Input: A JSON string with `latitude` (float) and `longitude` (float). "
                    f"You MUST include `\"current_weather\": true` to get the current conditions. "
                    f"Example: '{{\"latitude\": 37.77, \"longitude\": -122.41, \"current_weather\": true}}'." 
                )
            )
    raise ValueError("'/v1/forecast' GET endpoint not found in OpenAPI spec.")


def main():
    if not GOOGLE_API_KEY:
        logger.error("Error: GOOGLE_API_KEY environment variable not set. Please set it.")
        sys.exit(1)

    # 1. Configure LLM (Gemini)
    logger.info("Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_TO_USE,
        temperature=0.0, # Keep temperature at 0 for determinism
        google_api_key=GOOGLE_API_KEY
    )
    logger.info("Gemini LLM initialized.")

    # 2. Load Weather OpenAPI spec into a dictionary
    logger.info(f"Attempting to load and parse Weather spec from {WEATHER_SPEC_PATH} with PyYAML...")
    weather_spec_dict = {}
    try:
        with open(WEATHER_SPEC_PATH, "r") as f:
            weather_spec_dict = yaml.safe_load(f) 
        logger.info("Weather OpenAPI spec loaded into dictionary successfully.")
    except FileNotFoundError:
        logger.error(f"Error: Weather spec file not found at {WEATHER_SPEC_PATH}. Please ensure it exists.")
        sys.exit(1)
    except yaml.YAMLError as e: 
        logger.error(f"Error: Failed to parse YAML from {WEATHER_SPEC_PATH}: {e}. Ensure it's valid YAML.", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load or parse Weather spec from {WEATHER_SPEC_PATH} with PyYAML: {e}", exc_info=True)
        sys.exit(1)

    # 3. Create requests wrapper
    logger.info("Creating RequestsWrapper...")
    requests_wrapper = RequestsWrapper(headers={}) 
    logger.info("RequestsWrapper created.")

    # 4. Create tools (ONLY the weather forecast tool)
    logger.info("Creating weather forecast tool...")
    try:
        weather_forecast_tool = create_weather_forecast_tool_from_dict(weather_spec_dict, requests_wrapper)
        all_tools = [weather_forecast_tool] # Only one tool
        logger.info(f"Successfully created {len(all_tools)} tool(s).")
    except Exception as e:
        logger.error(f"Failed to create tools: {e}", exc_info=True)
        sys.exit(1)

    # 5. Create AgentExecutor
    logger.info("Creating AgentExecutor...")
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly specialized AI assistant for weather forecasts.
            You have access to ONE tool: `get_v1_forecast`.
            You MUST use the `get_v1_forecast` tool to answer questions about weather.
            The `get_v1_forecast` tool requires `latitude` and `longitude`.
            Always include `\"current_weather\": true` in the tool call to get current conditions.
            After calling the tool, summarize the weather information clearly and concisely from its output.
            Do NOT try to answer based on your internal knowledge. If you cannot get weather, inform the user clearly.
            """), # Prompt updated for single tool
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

    # --- Test the Agent with a direct coordinates query ---
    print("\n--- Agent Test with Weather OpenAPI ---")
    
    # Direct query with coordinates
    query = "What is the current weather at latitude 37.77 and longitude -122.41?" 
    print(f"\n--- Testing with query: '{query}' ---")
    try:
        logger.info(f"Invoking agent with query: '{query}'")
        result = agent_executor.invoke({"input": query}, stream_log=True) # Keep stream_log=True
        
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

        print("\nFinal Agent Response:")
        print(result.get('output', 'No output found.'))

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        print(f"An error occurred during agent execution: {e}. Please check logs for details.")

if __name__ == "__main__":
    main()