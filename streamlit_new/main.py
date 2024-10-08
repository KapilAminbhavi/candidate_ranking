import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
#from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time
from langchain.chat_models import ChatOpenAI
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
#groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Job Description Matcher")

llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=15000, temperature=0.2)

prompt = ChatPromptTemplate.from_template(
"""
You are an expert in analyzing resumes, with profound knowledge in technology, software engineering, data science, full stack web development, cloud engineering, 
cloud development, devops engineering, and big data engineering. 
Your role involves evaluating resumes against job descriptions.
Recognizing the competitive job market, provide top-notch assistance over the analysis of the resumes against the job description.

1. **Keyword Match Score**:
   - **Semantic & Contextual Relevance**: Provide the breakdown of how the semantic similarity and contextual relevance were calculated, including specific technical terms found in the resume and their relevance to the job description.
   - **Keyword Density**: Discuss the frequency and distribution of important keywords in the resume and how it impacts the overall keyword match score.

2. **Skill Experience Match Score**:
   - **Experience & Skill Level Alignment**: Break down the relevance of the candidate's past experience and proficiency levels in comparison to the job description's requirements, citing specific roles or projects.
   - **Recency & Skill Utilization Frequency**: Highlight the weight given to recent experience with key skills and how often and in what contexts the candidate has used the required skills, affecting the match score.

Also, provide a list of the candidate's strongest skills in order of relevance, and compare this order with the priority of skills in the job description. 

**Important Instructions:**
1. Display ALL candidates who have an overall match score of 70% or above.
2. Do not limit the number of candidates displayed.
3. Ensure that the skill experience match is given the highest priority when determining the overall match score.
4. The highest overall match should be reflected accurately in the Ideal Candidate Insight.
5. Make sure to extract and display the candidate's name from each resume. If the name is not clearly stated, use "Unnamed Candidate" followed by a unique identifier.

Today's date is 19th of August, 2024.

Job Description:
{input}

Resume Context:
{context}

The final response will strictly be in the following format for EACH candidate with a 70% or higher overall match:

Candidate: [Name or "Unnamed Candidate X" if name not found]
Keyword Match: [Percentage and a one line explanation]
Skill Experience Match: [Percentage and a one line explanation]
Prominent Skills: [Highlight the most prominent skills for the respective job description in one line.]
Overall Match: [Percentage]

After listing all matching candidates, provide:

Ideal Candidate Insight: [Provide a comprehensive analysis of the best candidate's resume among all in relation to the specific job description in just 2 lines and not more than that. Highlight the candidate's key strengths, relevant experiences, and overall fit for the role. Include an evaluation of how well the candidate's skills, experiences, and accomplishments align with the job requirements.]

If no candidates match 60% or above, state: "No candidates meet the 70% overall match threshold for this job description."
"""
)


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        resume_folder_path = os.path.join(os.path.dirname(__file__), 'hello')
        st.session_state.loader = PyPDFDirectoryLoader(resume_folder_path)
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        if st.session_state.final_documents:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        else:
            st.error("No documents were split for vectorization.")
    else:
        st.error("No documents were loaded from the resume folder.")

job_description = st.text_area("Enter the Job Description", height=100)

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if job_description:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': job_description})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
