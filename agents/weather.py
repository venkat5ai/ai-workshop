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
from langchain_community.utilities.requests import RequestsWrapper 
from langchain.tools import Tool 
from langchain_core.messages import HumanMessage

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
                print(f"\n--- DEBUG: Inside _dynamic_api_call for {tool_name} ---")
                print(f"--- DEBUG: Received JSON args: {json_args_str} ---")

                try:
                    # The LLM is passing a JSON string that might need to be parsed
                    # twice if it includes "__arg1" from the binding.
                    # Let's handle the __arg1 wrapper if it exists.
                    llm_args = json.loads(json_args_str)
                    if '__arg1' in llm_args:
                        args = json.loads(llm_args['__arg1'])
                    else:
                        args = llm_args
                except json.JSONDecodeError:
                    print("--- DEBUG: JSONDecodeError ---")
                    return f"Error: Invalid JSON input. Please provide parameters as a valid JSON string. Example: '{{\"latitude\": 37.77, \"longitude\": -122.41, \"current_weather\": true}}'"
                except Exception as e:
                    print(f"--- DEBUG: Error processing LLM args: {e} ---")
                    return f"Error processing LLM args: {e}. Input was: {json_args_str}"


                query_params = args
                full_url = f"{base_url}{path}" 

                logger.info(f"Making Weather API call: GET {full_url} with query {query_params}")
                
                try:
                    # --- CRITICAL FIX: Expect RequestsWrapper.get() to return text directly ---
                    # It means you should NOT call .raise_for_status() or .text on the result.
                    # The `RequestsWrapper` in this version directly returns response.text
                    response_text = requests_wrapper.get( 
                        url=full_url,
                        params=query_params, 
                    )
                    
                    # You would typically check HTTP status here with an actual response object.
                    # Since we only have the text, we'll assume success if JSON parsing works.
                    # For robust error handling, you'd want to check if the response_text 
                    # indicates an API error before trying to parse.
                    
                    # Check if response_text is empty or obviously an error message
                    if not response_text.strip():
                        print("--- DEBUG: Empty response text from API. ---")
                        return "Error: Empty response from weather API."
                    if "error" in response_text.lower() and "message" in response_text.lower():
                        print(f"--- DEBUG: API returned an error message: {response_text[:100]} ---")
                        return f"Weather API error: {response_text}" # Return the raw error message if it looks like one

                    # Try to parse the response text as JSON
                    parsed_response = json.loads(response_text)
                    print(f"--- DEBUG: API call successful, returning JSON. ---")
                    
                    # Return the JSON as a string for the LLM
                    return json.dumps(parsed_response) 

                except json.JSONDecodeError as e:
                    print(f"--- DEBUG: Failed to parse API response as JSON: {e} ---")
                    return f"Error: Could not parse API response as JSON: {e}. Raw response: {response_text[:200]}"
                except Exception as e:
                    logger.error(f"Error calling Weather API for {tool_name} ({summary}): {e}", exc_info=True)
                    print(f"--- DEBUG: API call failed: {e} ---")
                    return f"Error executing API call '{tool_name}': {e}. Input was: {json.dumps(args)}. Details: {e}"
            
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
        temperature=0.0, 
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
        all_tools = [weather_forecast_tool] 
        logger.info(f"Successfully created {len(all_tools)} tool(s).")
    except Exception as e:
        logger.error(f"Failed to create tools: {e}", exc_info=True)
        sys.exit(1)

    # --- Direct LLM Tool Calling Test (Bypass AgentExecutor for diagnosis) ---
    print("\n--- Direct LLM Tool Calling Test ---")
    
    # 1. Bind tools to the LLM to make it a tool-calling LLM
    tool_calling_llm = llm.bind_tools(all_tools)

    # 2. Define the message to send to the tool-calling LLM
    messages = [
        HumanMessage(
            content="What is the current weather at latitude 37.77 and longitude -122.41?"
        )
    ]

    # 3. Invoke the tool-calling LLM
    logger.info("Invoking tool-calling LLM directly with the weather query...")
    try:
        response = tool_calling_llm.invoke(messages)
        print("\n--- Direct LLM Response (for tool calls) ---")
        print(response) 

        if response.tool_calls:
            print("\nLLM successfully generated tool calls! Now executing them...")
            for tool_call in response.tool_calls:
                # Access dictionary keys instead of attributes
                print(f"  Tool Name: {tool_call['name']}")
                print(f"  Tool Args: {tool_call['args']}")
                
                # Manually execute the tool call for demonstration
                # The arguments from the LLM are also a dictionary
                if tool_call['name'] == "get_v1_forecast":
                    # Pass the args dictionary directly to the tool's func, it will handle json.dumps
                    # We also need to handle the __arg1 wrapping from the LLM's output
                    tool_output = weather_forecast_tool.func(json.dumps(tool_call['args']))
                    print(f"  Tool Output: {tool_output}")
                else:
                    print(f"  Warning: Unexpected tool call: {tool_call['name']}")
        else:
            print("LLM did NOT generate any tool calls for the direct query.")
            print(f"LLM Response content: {response.content}")

    except Exception as e:
        logger.error(f"Direct LLM tool calling test failed: {e}", exc_info=True)
        print(f"An error occurred during direct LLM test: {e}. Please check logs for details.")


if __name__ == "__main__":
    main()