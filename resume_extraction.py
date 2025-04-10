import os
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

# Folder path
folder_path = r"D:\Maran\Project\A Gen AI Sprint\Dataset\CVs1"
# Loop through all PDF files in the folder
pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
pdf_files.sort()
for filename in pdf_files[:2]:
    file_path = os.path.join(folder_path, filename)
    file = PdfReader(file_path)
    read = "\n".join([i.extract_text() for i in file.pages])

    # print(f"\n---- Contents of {filename} ----")
    # print(read[:200])
    
sections_to_extract = [
    "gmail ID",
    "Role Name",
    "work experience",
    "Skills"]

example_output = """
Example format:
Role - Data Scientist

Gmail ID - example@gmail.com

Work Experience - 
Data Scientist at ABC Inc. (2019-2023)
Built predictive models that enhanced decision-making processes, reducing operational costs by 25%.

Skills - 
Cybersecurity - Skilled in penetration testing, risk assessment, and securing enterprise networks against cyber threats.

"""