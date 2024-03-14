import streamlit as st
import os
import os.path

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, ServiceContext
from llama_index.core.storage import StorageContext
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI

load_dotenv()
storage_path = "./vectorstore"
documents_path = "./documents"

llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)


