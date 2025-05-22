# Ensure necessary libraries are installed
# You already installed these in the previous cells, but including for completeness
# !pip install PyPDF2 transformers accelerate datasets torch langchain

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import PyPDF2 or PyMuPDF if you need to extract text from a new PDF for testing
from PyPDF2 import PdfReader
import fitz # Import PyMuPDF (fitz) if you used the fallback logic

# Re-define text extraction functions if they are not already in the current cell
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2, with a fallback to PyMuPDF
    if PyPDF2 fails or yields no text. (Assuming these functions are copied or available)
    """
    text = ""
    print(f"Attempting to extract text from {pdf_path} using PyPDF2...")
    pypdf2_successful = False
    pypdf2_text = ""

    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            print(f"PyPDF2 reports PDF is encrypted: {pdf_path}. Skipping PyPDF2.")
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


# --- Load the Custom Trained Model ---
model_save_directory = "../data/models/my_insurance_model"

if not os.path.exists(model_save_directory):
    print(f"Error: Model directory '{model_save_directory}' not found.")
    print("Please ensure you have run the training code successfully to save the model.")
else:
    print(f"Loading model and tokenizer from '{model_save_directory}'...")
    try:
        # Load the tokenizer and model from the saved directory
        tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
        model = AutoModelForSequenceClassification.from_pretrained(model_save_directory)

        # Determine the device to use (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Set the model to evaluation mode
        print(f"Model loaded successfully to device: {device}")

        # --- Example Inference (Classifying New Text) ---

        print("\n--- Performing Inference ---")

        # You can test with a string directly
        test_text_string = "This document outlines the coverage details for my residential property at 123 Main St."

        # Or, extract text from a new PDF file for testing
        # Make sure this PDF file exists and replace 'path/to/your/new_test_document.pdf'
        test_pdf_path = '../data/insurance_pdfs/test.pdf' # Replace with a real path if testing a PDF
        test_text_from_pdf = ""
        if os.path.exists(test_pdf_path):
            print(f"\nExtracting text from test PDF: {test_pdf_path}")
            test_text_from_pdf = extract_text_from_pdf(test_pdf_path)
            if not test_text_from_pdf:
                 print(f"Warning: Could not extract text from {test_pdf_path}. Using string input instead.")
                 test_text_to_classify = test_text_string
            else:
                 test_text_to_classify = test_text_from_pdf
                 print("Using text extracted from PDF for classification.")
        else:
            print(f"\nTest PDF not found at {test_pdf_path}. Using string input for classification.")
            test_text_to_classify = test_text_string


        if not test_text_to_classify.strip():
            print("\nError: No text available to classify. Please provide text or a valid PDF path.")
        else:
            # Tokenize the input text
            inputs = tokenizer(test_text_to_classify, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            # Perform inference
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = model(**inputs)

            # Get the predicted class probabilities or logits
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1) # Convert logits to probabilities

            # Get the predicted class ID
            predicted_class_id = torch.argmax(logits, dim=1).item()

            # Map the predicted class ID back to the label string
            # The model object loaded from the saved directory should have the id2label mapping
            predicted_label = model.config.id2label[predicted_class_id]

            print(f"\nInput Text (partial):")
            print(test_text_to_classify[:500] + "...") # Print first 500 characters

            print(f"\nPredicted Label ID: {predicted_class_id}")
            print(f"Predicted Label: {predicted_label}")
            print(f"Class Probabilities: {probabilities.cpu().numpy()}") # Print probabilities for all classes


        print("\nInference process complete.")

    except Exception as e:
        print(f"\nError during model loading or inference: {e}")
        print("Please ensure the model directory contains the correct files and that libraries are installed.")


# Note: You cannot use Ollama directly with this model because it is a classification model,
# not a generative language model designed for conversational interaction.
# You use the 'transformers' library directly in Python for inference as shown above.