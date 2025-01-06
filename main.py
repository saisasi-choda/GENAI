from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import TextLoader
import json

pc = Pinecone(api_key='pcsk_445wgD_SPgH77uNYk6DbXBuHHCNWNpCrioCsnVJQtMS5UTeaUkr361fVrmfMNAhYs6GUCf')


loader=TextLoader(file_path="state_of_the_union.txt",encoding='utf8')
state_of_union=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=88)
text_chunks=text_splitter.split_documents(documents=state_of_union)
#print(len(chunks)) 42 


genai.configure(api_key=os.environ["GEMINI_API_KEY"])
chunk_embeddings=[]
for chunk in text_chunks:
    pre_embed_chunk=chunk
    result = genai.embed_content(model="models/text-embedding-004",content=f"{pre_embed_chunk}")
    embedding =(result['embedding'])
    chunk_embeddings.append(embedding)
    
    

index_name = "stateofunion1"

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
data_to_upsert = [
    (f"chunk-{i}", embedding, {"text": chunk.page_content})
    for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings))
]
print(data_to_upsert[:3])
# Insert into Pinecone
index.upsert(vectors=data_to_upsert)



def fetch_relevant_info(query):
    # Step 1: Generate query embedding
    query_result = genai.embed_content(model="models/text-embedding-004", content=query)
    query_embedding = query_result['embedding']  # Extract the embedding vector

    # Step 2: Query Pinecone using the embedding
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    relevant_data = "\n\n".join([match['metadata']['text'] for match in results['matches']])

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
        template="Based on the context: {context}\n\nGenerate a concise response (2 sentences) to the query: {query}"
    )

    chain = LLMChain(llm=llm, prompt=rag_prompt)

    response = chain.run({"context": relevant_data, "query": query})
    return response

query = "how much amount are we giving  direct assistance to Ukraine?"
response = fetch_relevant_info(query)
print(response)

    
        
    
        
    




