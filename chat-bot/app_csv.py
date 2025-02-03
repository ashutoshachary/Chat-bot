import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import glob

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_csv_text(csv_path):
    """Read and convert CSV content to text format"""
    try:
        df = pd.read_csv(csv_path)
        # Convert DataFrame to a string representation
        text = f"File: {os.path.basename(csv_path)}\n\n"
        # Add column descriptions
        text += "Columns: " + ", ".join(df.columns) + "\n\n"
        # Add the actual data
        text += df.to_string(index=False) + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error reading {csv_path}: {str(e)}")
        return ""


def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunk size for CSV data
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create and save vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Create the conversational chain with CSV-specific prompt"""
    try:
        prompt_template = """
        You are a data analysis assistant. Using the provided CSV data context, answer the question as detailed as possible.
        If you're performing calculations or analysis, explain your methodology.
        If the answer cannot be determined from the provided context, say "I cannot answer this question based on the available CSV data."

        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            max_retries=5,
            timeout=60,
        )

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    except Exception as e:
        st.error(f"Error initializing chat model: {str(e)}")
        st.warning("Please check your API quota or try again later.")
        return None


def user_input(user_question):
    """Process user questions and generate responses"""
    try:
        if not os.path.exists("faiss_index"):
            st.error("Please process the CSV files first before asking questions.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        if chain is None:
            return

        response = chain.invoke(
            {"input_documents": docs, "question": user_question}
        )

        st.write("Reply: ", response["output_text"])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("If this is a quota error, please wait a few minutes before trying again.")


def process_csv_folder():
    """Process all CSV files in the specified folder"""
    csv_folder = "data"  # folder name containing CSVs

    # Create data folder if it doesn't exist
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        st.warning("Created new 'data' folder. Please put your CSV files in it.")
        return False

    # Check for CSV files
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        st.error(f"No CSV files found in the '{csv_folder}' folder.")
        return False

    st.write("Found CSV files:", [os.path.basename(csv) for csv in csv_files])

    combined_text = ""
    for csv_path in csv_files:
        with st.spinner(f"Processing {os.path.basename(csv_path)}..."):
            combined_text += get_csv_text(csv_path)

    if not combined_text.strip():
        st.error("No valid data could be extracted from the CSV files.")
        return False

    with st.spinner("Creating text chunks..."):
        text_chunks = get_text_chunks(combined_text)

    with st.spinner("Building FAISS index..."):
        get_vector_store(text_chunks)

    return True


def main():
    st.set_page_config("Multi CSV Chatbot", page_icon=":bar_chart:")
    st.header("Multi-CSV 📊 - Chat Agent 🤖")

    user_question = st.text_input("Ask a question about your CSV data... ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")

        st.title("📊 CSV File's Section")
        st.write("CSVs will be loaded from the 'data' folder")

        if st.button("Process CSV Files"):
            with st.spinner("Processing CSVs from data folder..."):
                success = process_csv_folder()
                if success:
                    st.success("Successfully processed all CSV files!")

        st.write("---")
        st.write("AI App created by Ashutosh")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © Ashutosh
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()