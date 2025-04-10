import pandas as pd
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

data = pd.read_csv(r"D:\Maran\Project\A Gen AI Sprint\Dataset\job_description.csv", encoding='ISO-8859-1')
data = data.iloc[:2, :]
data['work_experience'] , data['skills'], data['role_embeddings'], data['description_embeddings'] = '', '','', ''
data['work_experience_embeddings'] ,data['skills_embeddings'] ='', ''

df = data.to_dict(orient='records')
df_role = [i['Job Title'] for i in df]
df_jd = [i['Job Description'] for i in df]

sections_to_extract = ["work experience", "Skills"]

work_experience_list, skills_list, work_experience_embeddings_list, skills_embeddings_list = [], [], [], []
role_embeddings_list, description_embeddings_list = [], []

example_output = """
Example format:
Role - Data Scientist

Work Experience - 
Data Scientist at ABC Inc. (2019-2023)
Built predictive models that enhanced decision-making processes, reducing operational costs by 25%.

Skills - 
Cybersecurity - Skilled in penetration testing, risk assessment, and securing enterprise networks against cyber threats.
"""

api_key = "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"
llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
model = SentenceTransformer('all-MiniLM-L6-v2')

for role, jd in zip(df_role[:2], df_jd[:2]):
    role_embeddings_list.append(model.encode(role).astype(np.float32).tobytes())
    description_embeddings_list.append(model.encode(jd).astype(np.float32).tobytes())
    
    for section in sections_to_extract:
        prompt = f"""
    Extract only the **{section}** section from the following JD. Use the format shown in the example.

    JD:
    {jd}

    {example_output}
    """
        response = llm.invoke(prompt).content
        
        if section == "work experience":
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
data.to_sql("jobs", conn, if_exists="replace", index=False)
# conn.close()
print("✅ Data saved to SQLite!")
