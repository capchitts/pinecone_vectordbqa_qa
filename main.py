import os
from dotenv import load_dotenv


from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone


pinecone.init(
    api_key="27f3f288-81f8-4561-ac68-a956dcc6f8b7", environment="us-east1-gcp"
)
load_dotenv()


if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("./mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    #OpenAI object to embedd the given text
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # put text and embedding object in the database
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blog-analyzer-project"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        vectorstore=docsearch,
        return_source_documents=True,
    )
    query = "What is a Vector database , give me a 15 words answer for a beginner"

    result = qa({"query": query})
    print(result)
