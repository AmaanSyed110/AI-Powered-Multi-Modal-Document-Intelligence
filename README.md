# AI-Powered Multi-Modal Document Intelligence

## Overview
AI-Powered Multi-Modal Document Intelligence is a Streamlit-based application that leverages OpenAI's GPT-4 and LangChain to provide a multi-modal Retrieval-Augmented Generation (RAG) system. This tool allows users to upload PDF documents, extract text, tables, and images, and interact with an AI-powered chatbot to ask questions and retrieve insights from the documents.

## Flow of the Project
![diagram-export-3-22-2025-5_08_33-PM](https://github.com/user-attachments/assets/66981af9-c0b5-4ca9-86e2-6e67df9bb67d)


## Features
- **Multi-Modal Processing**:
  - Extract text and tables from PDFs using ``pdfplumber``.
  - Extract and summarize images from PDFs using ``PyMuPDF`` and ``pdf2image``.
  - Compress and process images for efficient summarization.
 
- **AI-Powered Summarization**:
  - Summarize images using OpenAI's GPT-4 API.
  - Generate concise descriptions of images for better context.
 
- **Vector-Based Retrieval**:
  - Use OpenAI embeddings to create a FAISS vector store for efficient similarity search.
  - Combine text, tables, and image summaries into a unified vector space.
 
- **Conversational AI**:
  - Interact with the documents using a conversational retrieval chain powered by GPT-4.
  - Maintain chat history for context-aware responses.
 
- **Batch Processing**:
  - Process multiple PDFs in parallel using threading.
  - Handle large documents efficiently with batch summarization and rate limiting.
 
- **User-Friendly Interface**:
  - Built with Streamlit for an intuitive and interactive experience.
  - Upload PDFs, process them, and ask questions in real-time.
 
## Tech Stack
- **Streamlit**: For building the interactive web application.

- **OpenAI**: For text and image summarization using GPT-4.

- **LangChain**: For creating the conversational retrieval chain and managing embeddings.

- **FAISS**: For efficient similarity search in the vector store.

- **PyMuPDF and pdfplumber**: For extracting text, tables, and images from PDFs.

- **Pillow (PIL)**: For image processing and compression.

## Example Workflow
- Upload a PDF containing text, tables, and images.
  
- The application extracts:
  - Text and tables using ``pdfplumber``.
  - Images using ``PyMuPDF`` and ``pdf2image``.
  
- Images are summarized using OpenAI's GPT-4 API.
  
- Text, tables, and image summaries are combined into a FAISS vector store.

- Ask questions like:
  - "What is the summary of the document?"
  - "Can you describe the images in the document?"
  - "What are the key points in the tables?"

## Steps to run the AI-Powered Multi-Modal Document Inteligence in your system
- ### Clone the Repository
Open a terminal and run the following command to clone the repository:

```
git clone https://github.com/AmaanSyed110/AI-Powered-Multi-Modal-Document-Intelligence.git
```
- ### Set Up a Virtual Environment
It is recommended to use a virtual environment for managing dependencies:

```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
- ### Install Dependencies
Install the required packages listed in the ```requirements.txt``` file
```
pip install -r requirements.txt
```
- ### Add Your OpenAI API Key
Create a ```.env``` file in the project directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```
- ### Run the Application
Launch the Streamlit app by running the following command:
```
streamlit run app.py
```
- ### Upload PDF Documents
Use the sidebar to upload one or more PDF files.

- ### Process Documents
Click the "Process" button to extract text, tables, and images from the uploaded PDFs.

- ### Interact with the Application
Ask questions related to the PDFs, and the app will provide relevant responses based on the document content.

## Contributions
Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements or bug fixes.
