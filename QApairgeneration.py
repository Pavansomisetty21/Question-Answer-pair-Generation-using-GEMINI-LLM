import streamlit as st
import google.generativeai as genai
import json
import time
from PyPDF2 import PdfReader  # To read PDFs

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def generate_qa_pairs(text, num_pairs=3, format_output='plain', retry_attempts=3):
    """
    Generate question-answer pairs from a text input.

    Parameters:
    - text (str): The text to generate QA pairs from.
    - num_pairs (int): The number of QA pairs to generate.
    - format_output (str): The format for output ('plain' or 'json').
    - retry_attempts (int): Number of retry attempts in case of API failure.

    Returns:
    - str or dict: The generated QA pairs in the specified format.
    """
    if not isinstance(num_pairs, int) or num_pairs <= 0:
        raise ValueError("num_pairs must be a positive integer.")
    
    if format_output not in ['plain', 'json']:
        raise ValueError("format_output must be 'plain' or 'json'.")

    attempts = 0
    while attempts < retry_attempts:
        try:
            # Create the GenerativeModel instance
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Generate content with the specified prompt
            prompt = (f"Generate {num_pairs} question-answer pairs from the following text. "
                      f"Provide all questions first and then all answers, separated by new lines. "
                      f"Output in the form Q: question text\nA: answer text\n\nText: {text}")
            response = model.generate_content(prompt)

            # Clean up the generated content
            output = response.text.replace('*', '').replace('#', '')

            # Split the output into questions and answers
            lines = output.split('\n')
            questions = [line.replace('Q:', '').strip() for line in lines if line.startswith('Q:')]
            answers = [line.replace('A:', '').strip() for line in lines if line.startswith('A:')]

            # Ensure the number of questions and answers matches num_pairs
            if len(questions) > num_pairs:
                questions = questions[:num_pairs]
            if len(answers) > num_pairs:
                answers = answers[:num_pairs]

            # Format the output based on user preference
            if format_output == 'json':
                qa_json = [{'question': questions[i], 'answer': answers[i]} for i in range(len(questions))]
                output = json.dumps(qa_json, indent=2)
            else:  # plain text
                qa_text = "\n".join([f"Q{i+1}: {questions[i]}" for i in range(len(questions))]) + "\n\n"
                qa_text += "\n".join([f"A{i+1}: {answers[i]}" for i in range(len(answers))])
                output = qa_text

            return output

        except Exception as e:
            attempts += 1
            st.error(f"Attempt {attempts} failed: {e}")
            if attempts < retry_attempts:
                time.sleep(2)  # Wait before retrying

    return "Failed to generate QA pairs after several attempts."

# Streamlit App
def main():
    st.title("QA Pair Generator with Gemini")

    # API Key input
    apikey = st.text_input("Enter your API key:", type="password")
    if not apikey:
        st.warning("Please enter the API key to proceed.")
        return

    # Configure API key for Google Generative AI
    genai.configure(api_key=apikey)

    # Input method selection
    input_method = st.radio("Select input method:", ("Text", "PDF"))
    
    # Text input or PDF upload
    if input_method == "Text":
        text_input = st.text_area("Enter text:")
    else:
        pdf_file = st.file_uploader("Upload a PDF file:", type="pdf")
        if pdf_file:
            text_input = extract_text_from_pdf(pdf_file)
        else:
            text_input = ""

    # Additional options
    num_pairs = st.number_input("Number of QA pairs:", min_value=1, max_value=10, value=3)
    format_output = st.radio("Output format:", ("plain", "json"))

    # Generate QA pairs
    if st.button("Generate QA Pairs"):
        if text_input:
            with st.spinner("Generating QA pairs..."):
                qa_pairs_output = generate_qa_pairs(text_input, num_pairs, format_output)
                st.text_area("Output:", value=qa_pairs_output, height=300)
        else:
            st.warning("Please provide the input text or upload a PDF file.")

if __name__ == "__main__":
    main()
