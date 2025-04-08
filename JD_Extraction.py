import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


question = "what are the Responsibilities"

JD_list = """

Example
 Description:
We are seeking a skilled Software Engineer to design, develop, and maintain software applications. The ideal candidate will write efficient code, troubleshoot issues, and collaborate with teams to deliver high-quality solutions.

Responsibilities:
Develop, test, and deploy software applications.
Write clean, maintainable, and scalable code.
Collaborate with cross-functional teams to define and implement features.
Troubleshoot and debug issues for optimal performance.
Stay updated with emerging technologies and best practices.

Qualifications:
Bachelor's degree in Computer Science or a related field.
Proficiency in programming languages like Python, Java, or C++.
Experience with databases, web development, and software frameworks.
Strong problem-solving skills and attention to detail.
Ability to work both independently and in a team environment.
"""

prompt =  f"""
Extract the 'Qualifications' section from the following job description:
{JD_list}
"""
api_key = ""

llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash")
response = llm.invoke(prompt)
generate = response.content
print(generate)
