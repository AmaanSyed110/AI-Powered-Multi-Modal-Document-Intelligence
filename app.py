import io
import fitz  # PyMuPDF
import imageio
import logging
import threading
import base64
import streamlit as st
import pdfplumber
import os
import time
import tenacity
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError
from openai import OpenAI
from pdf2image import convert_from_bytes
from queue import Queue
from concurrent.futures import ThreadPoolExecutor


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the embedding model with OpenAI
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

# Initialize the LLM for conversational retrieval
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o",
    temperature=0.7
)

client = OpenAI(
    api_key=OPENAI_API_KEY
)

def compress_image(image, max_size=(800, 800), quality=85):
    """Compress and resize image to reduce size."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return buffer

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.info(f"Retrying in {retry_state.next_action.sleep} seconds...")
)
def summarize_single_image(image, client):
    """Summarize a single image with retry logic."""
    try:
        # Compress image
        compressed = compress_image(image)
        img_str = base64.b64encode(compressed.getvalue()).decode("utf-8")
        
        # Correct format for the vision API
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=150,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise descriptions of images."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        }
                    ]
                }
            ]
        )
        summary = response.choices[0].message.content
        logging.info(f"Successfully generated summary: {summary[:50]}...")
        return summary
    except Exception as e:
        logging.error(f"Error summarizing single image: {str(e)}")
        return "Unable to summarize this image."


def batch_summarize_images(images, client):
    """Process images in batches with rate limiting."""
    summaries = []
    batch_size = 2  # Process 2 images at a time
    delay = 10  # 10 second delay between batches
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_summaries = []
        
        logging.info(f"Processing batch {i//batch_size + 1} of {(len(images) + batch_size - 1)//batch_size}")
        
        for image in batch:
            try:
                summary = summarize_single_image(image, client)
                if summary != "Unable to summarize this image.":
                    batch_summaries.append(summary)
                    logging.info("Successfully added summary to batch")
                else:
                    logging.warning("Skipped an image due to summarization failure")
            except Exception as e:
                logging.error(f"Error in batch processing: {str(e)}")
                batch_summaries.append("Unable to summarize this image.")
        
        summaries.extend(batch_summaries)
        
        # If this isn't the last batch, wait before processing next batch
        if i + batch_size < len(images):
            logging.info(f"Waiting {delay} seconds before next batch...")
            time.sleep(delay)
    
    logging.info(f"Successfully processed {len(summaries)} image summaries")
    return summaries

def extract_images_from_pdf(pdf_docs):
    """Extract all images from uploaded PDF documents using multiple methods."""
    images = []
    for pdf in pdf_docs:
        try:
            logging.info(f"Processing file: {pdf.name}, Size: {len(pdf.getvalue())} bytes")
            pdf.seek(0)
            
            # Method 1: PyMuPDF
            try:
                pdf_file = fitz.open(stream=pdf.read(), filetype="pdf")
                for page_index in range(len(pdf_file)):
                    page = pdf_file[page_index]
                    image_list = page.get_images(full=True)
                    
                    for img in image_list:
                        xref = img[0]
                        base_image = pdf_file.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        try:
                            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            images.append(img_pil)
                            logging.info(f"PyMuPDF: Extracted image from page {page_index + 1}")
                        except Exception as e:
                            logging.error(f"PyMuPDF: Failed to process image: {str(e)}")
            except Exception as e:
                logging.error(f"PyMuPDF extraction failed: {str(e)}")
            
            # Method 2: pdfplumber
            if not images:
                pdf.seek(0)
                try:
                    with pdfplumber.open(pdf) as pdf_reader:
                        for i, page in enumerate(pdf_reader.pages):
                            for image in page.images:
                                try:
                                    img_bytes = image['stream'].get_data()
                                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    images.append(img_pil)
                                    logging.info(f"pdfplumber: Extracted image from page {i + 1}")
                                except Exception as e:
                                    logging.error(f"pdfplumber: Failed to process image: {str(e)}")
                except Exception as e:
                    logging.error(f"pdfplumber extraction failed: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error processing PDF {pdf.name}: {str(e)}")
            continue
        finally:
            pdf.seek(0)
    
    logging.info(f"Total images extracted: {len(images)}")
    return images

def summarize_image(image):
    """Summarize the content of an image using OpenAI's GPT-4o API."""
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Use OpenAI API to summarize the image
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes images."},
                {"role": "user", "content": f"data:image/png;base64,{img_str}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error summarizing image: {e}")
        return "Unable to summarize this image."

def get_pdf_text_and_tables(pdf_docs):
    """Extract text and tables from uploaded PDF documents."""
    text = ""
    tables = []
    try:
        for pdf in pdf_docs:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""  # Extract text
                    tables.extend(page.extract_tables())  # Extract tables
    except Exception as e:
        logging.error(f"Error extracting text and tables from PDF: {e}")
        st.error("Failed to extract text and tables from the PDF. Please check the file format.")
    return text, tables

def get_text_chunks(text):
    """Split text into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,  # Set chunk size to 1500 characters for optimal performance
        chunk_overlap=300,  # Overlap of 300 characters for better context retention
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, tables, image_summaries):
    """Create a FAISS vector store from text chunks, tables, and image summaries."""
    try:
        # Convert tables to text
        table_texts = ["Table: " + "\n".join(["\t".join(map(str, row)) for row in table]) for table in tables]
        
        # Combine all texts
        all_texts = text_chunks + table_texts + image_summaries
        
        # Batch process embeddings
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            logging.info(f"Processed embeddings batch {i//batch_size + 1}")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_embeddings(list(zip(all_texts, all_embeddings)), embeddings)
        logging.info("FAISS vector store created successfully.")
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error("Failed to create vector store. Please check the input data.")
        return None

def get_conversation_chain(vectorstore):
    """Create a conversation chain for the chatbot."""
    try:
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.7
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        st.error("Failed to create conversation chain. Please check the OpenAI API key.")
        return None
    
def process_pdf(pdf):
    """Process a single PDF and return its text, tables, and image summaries."""
    try:
        raw_text, tables = get_pdf_text_and_tables([pdf])
        images = extract_images_from_pdf([pdf])
        
        logging.info(f"Number of images extracted: {len(images)}")
        
        # Use batch processing for image summaries
        if images:
            image_summaries = batch_summarize_images(images, client)
        else:
            image_summaries = []
        
        logging.info(f"Number of image summaries generated: {len(image_summaries)}")
        
        return raw_text, tables, image_summaries
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return "", [], []

def handle_user_input(user_question):
    """Process user input and get response using the conversation chain."""
    if st.session_state.conversation:
        try:
            # Handle general queries using the conversation chain
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            # Display the updated chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"**User:** {message.content}")
                else:
                    st.write(f"**Assistant:** {message.content}")
            
            # Debug: Print retrieved documents
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    st.write(f"Retrieved document: {doc.page_content}")
        except Exception as e:
            logging.error(f"Error handling user input: {e}")
            st.error("An error occurred while processing your request. Please try again.")

def main():
    """Main application function."""
    st.set_page_config(page_title="Chat with Multi-Modal RAG", page_icon=":books:")
    st.title("Chat with Your Multi-Modal RAG :books:")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = {"text": "", "tables": [], "image_summaries": []}
    if "combined_text" not in st.session_state:
        st.session_state.combined_text = ""

    # User question input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs here:",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    try:
                        # Initialize variables to store combined data
                        all_text = ""
                        all_tables = []
                        all_image_summaries = []
                        
                        # Process each PDF in parallel using threading
                        threads = []
                        results = []
                        for pdf in pdf_docs:
                            thread = threading.Thread(target=lambda p=pdf: results.append(process_pdf(p)))
                            threads.append(thread)
                            thread.start()
                        for thread in threads:
                            thread.join()
                        
                        # Combine results
                        for result in results:
                            raw_text, tables, image_summaries = result
                            all_text += raw_text + "\n"
                            all_tables.extend(tables)
                            all_image_summaries.extend(image_summaries)
                        
                        if all_text.strip() == "" and not all_tables and not all_image_summaries:
                            st.error("No readable content found in the uploaded PDFs. Please check the PDFs.")
                        else:
                            # Process text chunks
                            text_chunks = get_text_chunks(all_text)
                            
                            # Create vector store
                            vectorstore = get_vectorstore(text_chunks, all_tables, all_image_summaries)
                            
                            # Update session state with processed data
                            st.session_state.processed_data = {
                                "text": all_text,
                                "tables": all_tables,
                                "image_summaries": all_image_summaries
                            }
                            st.session_state.combined_text = all_text
                            
                            # Create conversation chain
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("Documents processed successfully! You can now ask questions.")
                    except Exception as e:
                        logging.error(f"Error processing documents: {e}")
                        st.error("An error occurred while processing your documents. Please check the files and try again.")
            else:
                st.warning("Please upload at least one PDF to process.")

if __name__ == "__main__":
    main()