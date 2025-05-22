# Ensure necessary libraries are installed at the beginning
# !pip install pypdf PyMuPDF PyPDF2 transformers accelerate datasets torch langchain scikit-learn

import os
import pypdf # Keep the import for clarity and safety
from PyPDF2 import PdfReader
# Import specific exceptions from pypdf
from pypdf.errors import FileNotDecryptedError, WrongPasswordError
import fitz  # Import PyMuPDF
from langchain.schema import Document
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# --- Data Loading and Text Extraction ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2, with a fallback to PyMuPDF
    if PyPDF2 fails or yields no text.
    """
    text = ""
    print(f"Attempting to extract text from {pdf_path} using PyPDF2...")
    pypdf2_successful = False
    pypdf2_text = ""

    try:
        reader = PdfReader(pdf_path)

        if reader.is_encrypted:
            print(f"PyPDF2 reports PDF is encrypted: {pdf_path}. Skipping PyPDF2 page extraction.")
            pypdf2_successful = False
        else:
            for i, page in enumerate(reader.pages):
                try:
                    extracted_page_text = page.extract_text()
                    if extracted_page_text:
                        pypdf2_text += extracted_page_text + "\n"
                    else:
                        pass
                except Exception as page_e:
                    print(f"Warning: Error extracting text from page {i+1} in {pdf_path} using PyPDF2: {page_e}")
                    pass

            pypdf2_successful = True

    except FileNotDecryptedError:
        print(f"Error: PDF requires decryption but no password was provided for {pdf_path} (PyPDF2).")
        pypdf2_successful = False
    except WrongPasswordError:
         print(f"Error: Incorrect password provided for {pdf_path} (PyPDF2).")
         pypdf2_successful = False
    except Exception as e:
        print(f"An unexpected error occurred with PyPDF2 for {pdf_path}: {e}")
        pypdf2_successful = False

    if not pypdf2_successful or not pypdf2_text.strip():
         print(f"PyPDF2 extraction failed or yielded no text for {pdf_path}. Trying fallback method (PyMuPDF).")
         return extract_text_with_pymupdf(pdf_path)

    print(f"Successfully extracted text from {pdf_path} using PyPDF2.")
    return pypdf2_text.strip()


def extract_text_with_pymupdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF (fitz)."""
    text = ""
    print(f"Attempting to extract text from {pdf_path} using PyMuPDF...")
    try:
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted:
                print(f"PyMuPDF reports PDF is encrypted: {pdf_path}. Attempting decryption if possible...")
                if doc.is_encrypted:
                     print(f"Could not decrypt {pdf_path} with PyMuPDF.")
                     return ""

            for page in doc:
                try:
                    extracted_page_text = page.get_text("text")
                    if extracted_page_text:
                        text += extracted_page_text
                except Exception as page_e:
                    print(f"Error extracting text from a page in {pdf_path} using PyMuPDF: {page_e}")
                    pass

    except Exception as e:
        print(f"An unexpected error occurred with PyMuPDF for {pdf_path}: {e}")
        return ""

    if text.strip():
        print(f"Successfully extracted text from {pdf_path} using PyMuPDF.")
        return text.strip()
    else:
        print(f"PyMuPDF extracted no text from {pdf_path}.")
        return ""


# Assuming your PDF documents are in a directory named 'insurance_pdfs'
pdf_directory = '/content/insurance_pdfs'
insurance_documents = []

if not os.path.exists(pdf_directory):
    os.makedirs(pdf_directory)
    print(f"Directory '{pdf_directory}' created. Please place your PDF documents inside.")
else:
    print(f"Processing PDFs in directory: {pdf_directory}")
    if not os.listdir(pdf_directory):
        print(f"Directory '{pdf_directory}' is empty. Please place your PDF documents inside.")
    else:
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                extracted_text = extract_text_from_pdf(pdf_path)

                if extracted_text:
                    # --- START: UPDATED LABELING LOGIC BASED ON FILENAME PREFIX ---
                    label = "unknown" # Default label
                    filename_lower = filename.lower()

                    # Define a list of known prefixes and their corresponding labels
                    # YOU MUST ENSURE YOUR PDF FILENAMES USE THESE PREFIXES TO GET LABELS
                    label_prefixes = {
                        "home-": "home",
                        "condo-": "condo",
                        "vehicle-": "vehicle",
                        "auto-": "vehicle", # Add common synonyms
                        "property-": "property"
                        # Add any other prefixes you use for labeling
                    }

                    # Check if the filename starts with any of the known prefixes
                    for prefix, assigned_label in label_prefixes.items():
                        if filename_lower.startswith(prefix):
                            label = assigned_label
                            break # Stop after finding the first matching prefix

                    if label == "unknown":
                        print(f"Warning: Could not determine specific label for {filename} based on known prefixes. Assigning 'unknown'. This document will NOT be used for training.")
                        # Optional: print partial text to help identify how to label
                        # print("Partial text content for debugging:", extracted_text[:500] + "...")

                    # Only add documents with assigned labels (not 'unknown') to the list
                    if label != "unknown":
                         insurance_documents.append(Document(page_content=extracted_text, metadata={"source": pdf_path, "label": label}))
                         print(f"Assigned label '{label}' to document: {filename}")
                    else:
                         print(f"Skipping document '{filename}' from training data due to 'unknown' label.")

                    # --- END: UPDATED LABELING LOGIC ---

                else:
                     print(f"Skipping empty or unreadable PDF: {pdf_path}")


# Check if any *labeled* documents were loaded for training
labeled_data = [d for d in insurance_documents if d.metadata.get("label") != "unknown"]

if not labeled_data:
    print("\nNo documents with recognized labels available for training after applying labeling logic.")
    print("Please ensure your PDF filenames use one of the defined prefixes (e.g., 'home-', 'condo-', 'vehicle-').")
    print(f"Attempted to load {len(insurance_documents)} total documents, but none received a specific label.")
else:
    print(f"\nSuccessfully loaded {len(labeled_data)} documents with recognized labels for training.")
    print(f"Total documents processed (including 'unknown' or unreadable): {len(insurance_documents)}")

    # --- AI Model Training Steps (Supervised Text Classification Example) ---

    print("\nProceeding to supervised model training setup...")

    # 1. Prepare data for training (e.g., convert to Hugging Face Dataset)
    # We use 'labeled_data' which already excludes 'unknown'
    # Convert documents to a list of dictionaries suitable for Hugging Face Dataset
    processed_data_for_dataset = []
    for doc in labeled_data:
         label = doc.metadata.get("label")
         # We already filtered out 'unknown', so all documents here should have a valid label
         processed_data_for_dataset.append({"text": doc.page_content, "label": label})

    # Create a Hugging Face Dataset
    # In a real application, you would typically split this into train/validation/test datasets
    # For simplicity here, we use all labeled data as training data
    hf_dataset = Dataset.from_dict({"text": [d["text"] for d in processed_data_for_dataset],
                                    "label": [d["label"] for d in processed_data_for_dataset]})

    # For classification, you need a mapping from string labels to integer IDs
    unique_labels = list(set([d["label"] for d in processed_data_for_dataset]))
    # Ensure labels are sorted for consistent mapping
    unique_labels.sort()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)

    if num_labels < 2:
         print(f"\nWarning: Only {num_labels} unique label(s) found in the labeled data after filtering.")
         print(f"Labels found: {unique_labels}")
         print("Need at least 2 different labels for classification training. Cannot proceed with training.")
    else:
        # Update the dataset with integer labels
        def map_labels_to_ids(examples):
            return {"label": [label_map[label] for label in examples["label"]]}

        hf_dataset = hf_dataset.map(map_labels_to_ids, batched=True)

        print("\nHugging Face Dataset created from labeled data:")
        print(hf_dataset)
        print(f"Label mapping: {label_map}")


        # --- Tokenization, Model Loading, and Training Setup ---
        # Example: Text Classification using a pre-trained BERT model
        model_name = "bert-base-uncased" # Or another suitable model like roberta-base, etc.

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            # Tokenize the dataset
            # The Trainer expects the 'label' column to be present after tokenization
            def tokenize_function(examples):
               # We keep the 'label' column here
               return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512) # Adjust max_length

            # Apply tokenization and remove only the original text column
            tokenized_datasets = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])


            print("\nDataset after tokenization and adding integer labels:")
            print(tokenized_datasets)

            # Load the model for sequence classification
            id2label = {i: label for label, i in label_map.items()}
            label2id = label_map
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)

            if tokenizer.pad_token_id is not None and len(tokenizer) > model.get_input_embeddings().num_embeddings:
                 print(f"Resizing model token embeddings to match tokenizer size ({len(tokenizer)})...")
                 model.resize_token_embeddings(len(tokenizer))


            # Define Training Arguments (adjust these parameters for your training run)
            output_dir = "./results" # Directory to save checkpoints and outputs
            training_args = TrainingArguments(
                output_dir=output_dir,
                # Evaluation strategy is set to 'no' as we don't have a separate eval dataset
                # Corrected parameter name from evaluation_strategy to eval_strategy
                eval_strategy="no", # Use eval_strategy for transformers versions >= 4.2
                learning_rate=2e-5,
                per_device_train_batch_size=2, # Adjust based on GPU memory and dataset size
                per_device_eval_batch_size=2, # Set even if eval_strategy='no'
                num_train_epochs=3, # Start with a small number of epochs
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                save_steps=100, # Adjust based on dataset size
                save_total_limit=2,
                no_cuda=not torch.cuda.is_available(),
                report_to="none"
            )

            # Create the Trainer instance
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets, # Your tokenized training dataset
                # eval_dataset=... # Add an evaluation dataset here if you split your data
                tokenizer=tokenizer, # Pass the tokenizer
            )

            print("\nStarting model training...")
            # Start training - THIS IS WHERE THE MODEL GETS FINE-TUNED
            # This line will ONLY run if you have >= 2 labeled documents with >= 2 distinct labels
            trainer.train()
            print("Model training complete.")

            # --- Saving the Custom Trained Model Locally ---
            # THIS IS WHERE YOUR CUSTOM TRAINED MODEL IS SAVED
            save_directory = "./my_insurance_model"
            print(f"\nSaving custom trained model and tokenizer to {save_directory}...")
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            print(f"Custom trained model and tokenizer saved locally to '{save_directory}'.")
            print(f"To view the saved files, look for a directory named '{save_directory}' in the Colab file explorer (left sidebar).")
            print("This directory contains your fine-tuned model files (like pytorch_model.bin) and tokenizer files.")


            # --- Pushing to Hugging Face (Conceptual) ---
            # This part is commented out. To publish, you would uncomment this
            # and ensure you are authenticated with Hugging Face.
            # print(f"\nPushing custom trained model and tokenizer to Hugging Face Hub (conceptual)...")
            # try:
            #     # Ensure you are logged in: !huggingface-cli login in a new cell
            #     # Or programmatically: from huggingface_hub import notebook_login; notebook_login() in a new cell
            #     # Specify your repository name (e.g., "your-username/your-model-name")
            #     # model_repo_name = "your-huggingface-username/your-model-name"

            #     # model.push_to_hub(model_repo_name)
            #     # tokenizer.push_to_hub(model_repo_name)
            #     # print(f"Model and tokenizer pushed to Hugging Face Hub under {model_repo_name}.")
            # except Exception as e:
            #     print(f"Error pushing to Hugging Face Hub: {e}")
            #     print("Please ensure you are logged in, the repository name is valid, and the model is saved locally first.")


        except Exception as e:
            print(f"\nError during model setup or training: {e}")
            print("Please check your data, model configuration, and training arguments.")

print("\nProcess finished.")