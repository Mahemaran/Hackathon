import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from google import genai
from google.genai import types

data = pd.read_csv(r"D:\Maran\Project\A Gen AI Sprint\Dataset\job_description.csv", encoding='ISO-8859-1')
data = data.iloc[:2, :]
data['work_experience'] , data['skills'], data['role_embeddings'], data['description_embeddings'] = '', '','', ''
data['work_experience_embeddings'] ,data['skills_embeddings'] ='', ''

df = data.to_dict(orient='records')
df_role = [i['Job Title'] for i in df]
df_jd = [i['Job Description'] for i in df]

work_experience_list, skills_list, work_experience_embeddings_list, skills_embeddings_list = [], [], [], []
role_embeddings_list, description_embeddings_list = [], []

model = SentenceTransformer('all-MiniLM-L6-v2')
client = genai.Client(api_key="AIzaSyAHdrg4G4Ves-qh8GcNfwh0FiPIGRCsGic")

for role, jd in zip(df_role[:2], df_jd[:2]):
    role_embeddings_list.append(model.encode(role).astype(np.float32).tobytes())
    description_embeddings_list.append(model.encode(jd).astype(np.float32).tobytes())
    
    prompts = {
    "experience": f"""
Extract only the work experience from the following jd text: {jd}. 
""",

    "skills": f"""
Extract only the skills and tools from the following jd text: {jd}. 
"""
}
    
    for key, prompt in prompts.items() :
        responses = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0))
        response =responses.text.strip()
        
        if key == "experience":
            work_experience_list.append(response)
            work_experience_embeddings_list.append(model.encode(response).astype(np.float32).tobytes())
        else:
            skills_list.append(response)
            skills_embeddings_list.append(model.encode(response).astype(np.float32).tobytes())

data['work_experience'], data['skills'] = work_experience_list, skills_list
data['role_embeddings'], data['description_embeddings'] = role_embeddings_list, description_embeddings_list
data['work_experience_embeddings'] ,data['skills_embeddings'] = work_experience_embeddings_list, skills_embeddings_list

# Connect to SQLite (or create)
conn = sqlite3.connect("D:\Maran\Project\A Gen AI Sprint\Dataset\database\smarthire_db.db")
# Store in a new table called 'jobs'
data.to_sql("JD", conn, if_exists="replace", index=False)
conn.close()
print("✅ Data saved to SQLite!")
