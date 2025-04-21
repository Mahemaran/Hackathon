import os
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

role_list, work_experience_list, skills_list, relevance_list  = [], [], [], []
role_embeddings_list, work_experience_embeddings_list, skills_embeddings_list, relevance_embeddings_list = [], [], [], []

pdf_files = [f for f in os.listdir("D:\Maran\Project\A Gen AI Sprint\Dataset\CVs1") if f.lower().endswith('.pdf')]
pdf_files.sort()
for filename in pdf_files[:2]:
    file_path = os.path.join("D:\Maran\Project\A Gen AI Sprint\Dataset\CVs1", filename)
    file = PdfReader(file_path)
    read = "\n".join([i.extract_text() for i in file.pages])
    relevance_list.append(read)
    
model = SentenceTransformer('all-MiniLM-L6-v2')
client = genai.Client(api_key="AIzaSyAHdrg4G4Ves-qh8GcNfwh0FiPIGRCsGic")

for resume in relevance_list:

    relevance_embeddings_list.append(model.encode(resume).astype(np.float32).tobytes())

    prompts = {
    "role": f"""
Extract only the current job title (role name) from the following resume text: {resume}.
Provide just the latest role name, without any additional information or explanation.
Example: "Data Scientist"
""",

    "experience": f"""
Extract only the work experience from the following resume text: {resume}.
""",

    "skills": f"""
Extract only the skills and tools from the following resume text: {resume}.
"""
}
    for key, prompt in prompts.items() :
        responses = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0))
        response =responses.text.strip()
        print(f"key- {response}\n")

        if key == "role":
            role_list.append(response)
            role_embeddings_list.append(model.encode(response).astype(np.float32).tobytes())

        elif key == "experience":
            work_experience_list.append(response)
            work_experience_embeddings_list.append(model.encode(response).astype(np.float32).tobytes())

        else:
            skills_list.append(response)
            skills_embeddings_list.append(model.encode(response).astype(np.float32).tobytes())

df = pd.DataFrame({
    'Role': role_list,
    'Work Experience': work_experience_list,
    'Skills': skills_list,
    'Relevance': relevance_list,
    'Role Embeddings': role_embeddings_list,
    'Work Experience Embeddings': work_experience_embeddings_list,
    'Skills Embeddings': skills_embeddings_list,
    'Relevance Embeddings': relevance_embeddings_list})

# Connect to SQLite (or create)
conn = sqlite3.connect("D:\Maran\Project\A Gen AI Sprint\Dataset\database\smarthire_db.db")
# Store in a new table called 'jobs'
df.to_sql("Resume", conn, if_exists="replace", index=False)
conn.close()
print("✅ Data saved to SQLite!")
