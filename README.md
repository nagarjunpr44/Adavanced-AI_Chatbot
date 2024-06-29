# Adavanced-AI_Chatbot


This repository contains a Jupyter notebook that demonstrates how to create an AI chatbot using the Retrieval-Augmented Generation (RAG) stack with Astra DB and OpenAI's language model. The chatbot can retrieve context from a vector database and generate responses based on the provided context.

## Overview

This notebook covers the following steps:
1. Setting up the environment with necessary libraries.
2. Configuring the embedding model and vector store.
3. Loading and preparing a dataset of philosopher quotes.
4. Inserting documents into the vector store.
5. Performing basic retrieval and generating responses using a prompt.
6. Cleaning up the collection in the vector store.

## Requirements

- Python 3.x
- Jupyter Notebook
- Astra DB account and API credentials
- OpenAI API key

## Installation

To get started, clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/ai_chatbot_ragstack.git
cd ai_chatbot_ragstack
pip install -r requirements.txt
```

The `requirements.txt` file should include:
```
ragstack-ai
datasets
langchain-openai
cassio
```

Alternatively, you can install the packages directly in the notebook:

```python
!pip install -q ragstack-ai datasets
!pip install -U langchain-openai
```

## Usage

1. **Configure Environment Variables**: Enter your Astra DB API endpoint, application token, and OpenAI API key.

```python
import os
from getpass import getpass

os.environ["ASTRA_DB_API_ENDPOINT"] = "enter astra db api endpoint"
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = "enter astra db application token"
os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key: ")
```

2. **Create RAG Pipeline**: Configure the embedding model and vector store.

```python
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
vstore = AstraDBVectorStore(
    collection_name="test",
    embedding=embedding,
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
)
print("Astra vector store configured")
```

3. **Load Dataset**: Load a sample dataset of philosopher quotes.

```python
from datasets import load_dataset

philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
print("An example entry:")
print(philo_dataset[16])
```

4. **Prepare Documents**: Create a set of documents from the dataset.

```python
from langchain.schema import Document

docs = []
for entry in philo_dataset:
    metadata = {"author": entry["author"]}
    if entry["tags"]:
        for tag in entry["tags"].split(";"):
            metadata[tag] = "y"
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)
```

5. **Insert Documents**: Add the documents to the vector store.

```python
inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")
```

6. **Retrieve and Generate Response**: Use the model to generate responses based on retrieved context.

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

retriever = vstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("In the given context, what is happiness and can you explain more and who is the author?")
```

7. **Cleanup**: Delete the collection and all documents in the collection.

```python
vstore.delete_collection()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

