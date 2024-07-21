from fastapi import FastAPI, File, UploadFile, Form
from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
import logging
from bs4 import BeautifulSoup
import requests
from llama_index.llms.groq import Groq

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scrape/")
async def scrape_site(url: str = Form(...)):
    if url:
        response = requests.get(url)
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            tags = soup.find_all(['h1', 'h2', 'h3', 'p'])
            return ''.join(str(tag) for tag in tags)
        else:
            return ""
    return ""

@app.post("/query/")
async def query_file(prompt: str = Form(...), files: List[UploadFile] = File(...)):
    llm = Groq(model="llama3-8b-8192", api_key="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")
    Settings.llm = llm
    
    # Save the uploaded files to a temporary directory
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        logging.info(f"Received file: {file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    logging.info(f"Received prompt: {prompt}")
    
    # Load data from the saved files
    documents = SimpleDirectoryReader(temp_dir).load_data()
    
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    
    # Create index and query engine
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    # Perform the query with the given prompt
    response = query_engine.query(prompt)
    
    return {"response": response}


# To run the server, use the command: uvicorn main:app --reload
