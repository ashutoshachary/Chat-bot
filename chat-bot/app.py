import streamlit as st
from PyPDF2 import PdfReader
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
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # textembedding-gecko@001
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/textembedding-gecko@001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n
#
#     Answer:
#     """
#
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain


def get_conversational_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        # model = ChatGoogleGenerativeAI(model="gemini-pro",
        #                                temperature=0.3,
        #                                max_retries=2,  # Reduce retries
        #                                timeout=30)  # Add timeout
        # gemini - pro - vision
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                       temperature=0.3,
                                       max_retries=5,  # Increased from 2
                                       timeout=60,)

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    except Exception as e:
        st.error(f"Error initializing chat model: {str(e)}")
        st.warning("Please check your API quota or try again later.")
        return None


# def user_input(user_question):
#     if not os.path.exists("faiss_index"):
#         st.error("Please process the PDF files first before asking questions.")
#         return
#
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings)
#     docs = new_db.similarity_search(user_question)
#
#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": user_question}
#         , return_only_outputs=True)
#
#     st.write("Reply: ", response["output_text"])


def user_input(user_question):
    try:
        if not os.path.exists("faiss_index"):
            st.error("Please process the PDF files first before asking questions.")
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
# def process_docs_folder():
#     docs_folder = "docs"
#
#     # Create docs folder if it doesn't exist
#     if not os.path.exists(docs_folder):
#         os.makedirs(docs_folder)
#         st.warning("Created new 'docs' folder. Please put your PDF files in it.")
#         return False
#
#     # Check for PDF files
#     pdf_files = glob.glob(os.path.join(docs_folder, "*.pdf"))
#     if not pdf_files:
#         st.warning("No PDF files found in the 'docs' folder. Please add your PDFs and try again.")
#         return False
#
#     st.write("Found PDFs:", [os.path.basename(pdf) for pdf in pdf_files])
#
#     combined_text = ""
#     for pdf_path in pdf_files:
#         with st.spinner(f"Processing {os.path.basename(pdf_path)}..."):
#             combined_text += get_pdf_text(pdf_path)
#
#     with st.spinner("Creating text chunks..."):
#         text_chunks = get_text_chunks(combined_text)
#
#     with st.spinner("Building FAISS index..."):
#         get_vector_store(text_chunks)
#
#     return True


def process_docs_folder():
    docs_folder = "docs"  # folder name containing PDFs
    if not os.path.exists(docs_folder):
        st.error(f"The '{docs_folder}' folder doesn't exist in the project directory.")
        return False

    pdf_files = glob.glob(os.path.join(docs_folder, "*.pdf"))
    if not pdf_files:
        st.error(f"No PDF files found in the '{docs_folder}' folder.")
        return False

    combined_text = ""
    for pdf_path in pdf_files:
        combined_text += get_pdf_text(pdf_path)

    text_chunks = get_text_chunks(combined_text)
    get_vector_store(text_chunks)
    return True


def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDFs 📚 - Chat Agent 🤖 ")

    user_question = st.text_input("Ask a Question from the PDF Files... ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 PDF File's Section")
        st.write("PDFs will be loaded from the 'docs'")

        if st.button("Process PDF Files"):
            with st.spinner("Processing PDFs from docs folder..."):
                success = process_docs_folder()
                if success:
                    st.success("Successfully processed all PDFs from the docs folder!")

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