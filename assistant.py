# Ensure OpenAI library is installed
# !pip install openai python-dotenv tenacity

import os
import time
import random # Import random for jitter
from openai import OpenAI, APIError, RateLimitError
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt # No longer importing wait_random_jitter

# Load environment variables (including OPENAI_API_KEY) from a .env file if present
load_dotenv()

# --- Configure OpenAI Client ---
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key securely in your environment or in a .env file.")
    print("You can get your API key from https://platform.openai.com/api-keys")
    exit()

client = OpenAI(api_key=openai_api_key)
print("OpenAI client initialized.")


# --- Configuration for Reusing Resources ---
REUSE_EXISTING_ASSISTANT = True # <<< Set to True

# If REUSE_EXISTING_ASSISTANT is True, provide the IDs of your existing resources:
existing_assistant_id = None # <<< Set this to your existing Assistant ID if you have one you want to reuse
existing_vector_store_id = "vs_682ff230fb5881918ce583938808d599" # <<< Your existing Vector Store ID


# --- Custom Wait Function with Jitter ---
def wait_exponential_with_jitter(multiplier=1, min=1, max=60, jitter_range=(0, 2)):
    """
    Combines exponential backoff with random jitter.
    Returns a callable that tenactiy can use for the 'wait' parameter.
    """
    exponential_wait = wait_exponential(multiplier=multiplier, min=min, max=max)

    def _wait_with_jitter(retry_state):
        """The actual wait function called by tenacity."""
        base_wait_time = exponential_wait(retry_state) # Get the exponential wait time
        jitter = random.uniform(jitter_range[0], jitter_range[1]) # Generate random jitter
        return base_wait_time + jitter # Add jitter to the base wait time

    return _wait_with_jitter


# --- Exponential Backoff Decorator ---
@retry(
    # --- UPDATED: Use the custom wait function ---
    wait=wait_exponential_with_jitter(multiplier=1, min=1, max=60, jitter_range=(0, 2)),
    stop=stop_after_attempt(5),
    retry=(lambda retry_state: isinstance(retry_state.outcome.exception(), RateLimitError) or
           (isinstance(retry_state.outcome.exception(), APIError) and retry_state.outcome.exception().status_code in [408, 429, 500, 502, 503, 504]))
)
def call_openai_api_with_backoff(api_call_func, *args, **kwargs):
    """Helper function to wrap OpenAI API calls with exponential backoff."""
    return api_call_func(*args, **kwargs)


# --- Step 1: Upload Files (SKIPPED - Using existing Vector Store) ---
print("\n--- Skipping File Upload (Using existing Vector Store) ---")
uploaded_file_ids = []


# --- Step 2: Create or Update an Assistant with File Search ---
assistant_id = existing_assistant_id

if assistant_id and REUSE_EXISTING_ASSISTANT:
    print(f"\n--- Using existing Assistant with ID: {assistant_id} ---")
    try:
        assistant = call_openai_api_with_backoff(client.beta.assistants.retrieve, assistant_id)
        print("Retrieved existing assistant.")

        print(f"Ensuring Assistant {assistant_id} is linked to Vector Store {existing_vector_store_id}...")
        assistant = call_openai_api_with_backoff(
            client.beta.assistants.update,
            assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [existing_vector_store_id]}}
        )
        print("Assistant updated with correct Vector Store link.")

    except Exception as e:
        print(f"Error retrieving or updating existing assistant {assistant_id}: {e}")
        print("Falling back to creating a new assistant (and you should reconsider REUSE_EXISTING_ASSISTANT=True).")
        assistant_id = None


if not assistant_id:
    print("\n--- Creating a new Assistant with file_search tool (Fallback) ---")
    if REUSE_EXISTING_ASSISTANT:
        print("Warning: REUSE_EXISTING_ASSISTANT is True, but failed to retrieve existing assistant. Creating a new one.")

    try:
        assistant = call_openai_api_with_backoff(
            client.beta.assistants.create,
            name="HOA Financial Assistant (Created by Script)",
            instructions="You are an AI assistant specialized in answering questions about Home Owners Association (HOA) financial balance sheets. Use the file search tool to find specific transaction details, account balances, income, and expenses within the uploaded monthly financial statements.",
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [existing_vector_store_id]}}
        )
        assistant_id = assistant.id
        print(f"Assistant created with ID: {assistant_id}")
    except Exception as e:
        print(f"Error creating Assistant: {e}")
        print("Cannot proceed without a valid Assistant.")
        exit()


# --- Step 2.1: Create a Vector Store and Link Files (SKIPPED - Using existing Vector Store) ---
print("\n--- Skipping Vector Store Creation (Using existing Vector Store) ---")
vector_store_id = existing_vector_store_id
print(f"Using existing Vector Store with ID: {vector_store_id}")


# Give OpenAI's file search system some time to process the files in the vector store
# This sleep is less necessary when reusing an existing vector store where processing is likely done,
# but harmless. Remove if you are confident files are fully processed.
# print("Giving OpenAI's file search time to process files (waiting 10 seconds)...")
# time.sleep(10) # Reduced sleep time as files were likely processed manually


# --- Step 3: Create a Thread ---
print("\n--- Creating a new Conversation Thread ---")
try:
    thread = call_openai_api_with_backoff(client.beta.threads.create)
    thread_id = thread.id
    print(f"Thread created with ID: {thread_id}")
except Exception as e:
    print(f"Error creating Thread: {e}")
    if 'assistant' in locals() and assistant_id and not REUSE_EXISTING_ASSISTANT:
         try:
             client.beta.assistants.delete(assistant_id)
             print(f"Deleted Assistant {assistant_id} due to error.")
         except Exception as cleanup_e:
              print(f"Error during assistant cleanup: {cleanup_e}")
    exit()


# --- Step 4 & 5: Add a Message (the Question) and Run the Assistant ---
your_question = "Summarize the main income and expenses for both March and April."

print(f"\n--- Asking Question ---")
print(f"Question: {your_question}")

try:
    message = call_openai_api_with_backoff(
        client.beta.threads.messages.create,
        thread_id=thread_id,
        role="user",
        content=your_question,
    )
    print("Message added to thread.")

    run = call_openai_api_with_backoff(
        client.beta.threads.runs.create,
        thread_id=thread_id,
        assistant_id=assistant_id,
        max_completion_tokens=1000,
    )
    run_id = run.id
    print(f"Run created with ID: {run_id}. Status: {run.status}")

except Exception as e:
    print(f"Error adding message or creating Run: {e}")
    print(f"Error details: {e}")
    if 'thread' in locals() and thread_id:
         try:
              call_openai_api_with_backoff(client.beta.threads.delete, thread_id)
              print(f"Deleted Thread {thread_id} due to error.")
         except Exception as cleanup_e:
              print(f"Error during thread cleanup: {cleanup_e}")
    exit()


# --- Step 6: Poll the Run Status ---
print("Waiting for Run to complete...")
try:
    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(1)
        run = call_openai_api_with_backoff(
            client.beta.threads.runs.retrieve,
            thread_id=thread_id,
            run_id=run_id
        )
        print(f"Run status: {run.status}")
        if run.status == 'requires_action':
             print("Run requires action (e.g., function calling). This example does not handle required actions.")
             break

    if run.status == 'completed':
        print("Run completed successfully.")
    elif run.status == 'failed':
        print(f"Run failed.")
        print(f"Error: {run.last_error}")
    else:
        print(f"Run ended with status: {run.status}")

except Exception as e:
    print(f"Error polling Run status: {e}")
    print(f"Error details: {e}")
    if 'thread' in locals() and thread_id:
         try:
              call_openai_api_with_backoff(client.beta.threads.delete, thread_id)
              print(f"Deleted Thread {thread_id} due to error.")
         except Exception as cleanup_e:
              print(f"Error during thread cleanup: {cleanup_e}")


# --- Step 7: Retrieve Messages (Get Assistant's Response) ---
if run.status == 'completed':
    print("\n--- Retrieving Messages ---")
    try:
        messages = call_openai_api_with_backoff(
          client.beta.threads.messages.list,
          thread_id=thread_id,
          order="asc"
        )

        print("Assistant's Response:")
        assistant_messages_for_run = [
            msg for msg in messages.data if msg.run_id == run_id and msg.role == "assistant"
        ]

        if assistant_messages_for_run:
            for message in assistant_messages_for_run:
                for content_block in message.content:
                    if content_block.type == 'text':
                        print(content_block.text.value)
                        if content_block.text.annotations:
                            print("\nCitations:")
                            for annotation in content_block.text.annotations:
                                if annotation.type == 'file_citation':
                                    print(f"- File Citation: {annotation.text}")
                                    try:
                                        cited_file = client.files.retrieve(annotation.file_citation.file_id)
                                        print(f"  (Source File: {cited_file.filename})")
                                    except Exception as file_e:
                                        print(f"  (Could not retrieve file info for ID: {annotation.file_citation.file_id})")

                                elif annotation.type == 'file_path':
                                     print(f"- File Path Reference: {annotation.text}")

        else:
            print("No response found from the assistant for this run.")

    except Exception as e:
        print(f"Error retrieving messages: {e}")
        print(f"Error details: {e}")


elif run.status == 'failed':
     print("\nAssistant Run failed. Check the error details above.")
else:
     print("\nRun did not complete successfully. No messages retrieved for this run.")


# --- Optional Cleanup ---
print("\n--- Cleaning up (Deleting Thread if created in this run) ---")
try:
    if 'thread' in locals() and thread_id:
         print(f"Deleting Thread: {thread_id}...")
         call_openai_api_with_backoff(client.beta.threads.delete, thread_id)
         print("Thread deleted.")
except Exception as e:
    print(f"Error during cleanup: {e}")
    print("You may need to manually delete the Thread from the OpenAI platform if it persists.")


print("\nProcess finished.")