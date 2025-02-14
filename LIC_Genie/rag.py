
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langchain.schema.runnable import RunnableParallel
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import logging

logger = logging.Logger(__name__)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = "" # Disables gRPC tracing



pc = Pinecone(api_key=os.getenv('VECTOR_DB_API_KEY'))
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def document_loader(file_path):
    if file_path.endswith(".txt"):
        logger.info("inside document loader")
        loader=TextLoader(file_path=file_path,encoding='utf8')
        document=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=88)
        text_chunks=text_splitter.split_documents(documents=document)
        return text_chunks
        
    elif file_path.endswith(".pdf"):
        logger.info("inside document loader for pdf")
        text_chunks=[]
        loader=PyPDFLoader(file_path=file_path)
        pages = loader.load_and_split()
        for page in pages:
            text_chunks.append(page.page_content)
        return text_chunks
            
#print(len(chunks)) 42 

def get_embeddings(text_chunks):
    chunk_embeddings=[]
    for chunk in text_chunks:
        pre_embed_chunk=chunk
        result = genai.embed_content(model="models/text-embedding-004",content=f"{pre_embed_chunk}")
        embedding =(result['embedding'])
        print(f"embedding",embedding)
        chunk_embeddings.append(embedding)
    return chunk_embeddings

def store_in_vector_db(index_name,text_chunks,chunk_embeddings): 
    index_name = index_name
    index_names = [index['name'] for index in pc.list_indexes()]
    if index_name not in index_names:     
        pc.create_index(
            name=index_name,
            dimension=768, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )

    index = pc.Index(index_name)
    # Prepare data for upsert
    # data_to_upsert = [
    #     (f"chunk-{i}", embedding, {"text": chunk})
    #     for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings))
    # ]
    for i in range(len(text_chunks)):
        vectors=[(str(i),chunk_embeddings[i], {"text" : text_chunks[i]})]
        index.upsert(vectors=vectors)
    # print(data_to_upsert[:3])
    # Insert into Pinecone
    return True



def fetch_relevant_info(query,index_name):
    # Step 1: Generate query embedding
    query_result = genai.embed_content(model="models/text-embedding-004", content=query)
    query_embedding = query_result['embedding']  # Extract the embedding vector

    # Step 2: Query Pinecone using the embedding
    index_names = [index['name'] for index in pc.list_indexes()]
    if index_name in index_names:
        index = pc.Index(index_name)
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        relevant_data = "\n\n".join([match['metadata']['text'] for match in results['matches']])
    else:
        print(index_names)
        return "cannot list the indexes for the given name in pine cone"

    if not relevant_data:
        return "No relevant data found."

    # Step 3: Generate a concise response using the context
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ["GEMINI_API_KEY"]
    )

    rag_prompt = PromptTemplate(
        input_variables=["context", "query"],
        # template=""" imagine you are LIC agent bot context Based on the context: {context} \n\n Generate a concise response (2 sentences) to the query and explain the benifits: {query}by the customer"""
        template= """ You are an experienced LIC (Life Insurance Corporation) agent, providing clear and concise explanations about life insurance policies and related financial concepts. 
            Using the retrieved context below, answer the given query few sentences, ensuring simplicity and clarity for a potential customer. if the customer need to know the full detail give him important information like premium amount ,scheme name , tennure and so on 
            ### Context:  
            {context}  
            ### Query:  
            {query}  
            ### Response:"""
    )
    
    chain = rag_prompt | llm  # Runnable sequence
    response = chain.invoke({"context": relevant_data, "query": query})
    return response.content
def get_indices():
    index_names = [index['name'] for index in pc.list_indexes()]
    return index_names

def new_doc_pipeline(file_path,index_name):
    try:
        
        text_chunks=document_loader(file_path=file_path)
        chunk_embeddings=get_embeddings(text_chunks=text_chunks)
        flag=store_in_vector_db(index_name=index_name,text_chunks=text_chunks,chunk_embeddings=chunk_embeddings)
        if flag==True:
            return "document stored in vector db sucessfully"
        return None
    except Exception as e:
            print(e)
    
    
# if __name__=="__main__":
#     resp=new_doc_pipeline(file_path="./data/lic1.pdf",index_name="lic3")
#     print(resp)
    
 


    
        
    




