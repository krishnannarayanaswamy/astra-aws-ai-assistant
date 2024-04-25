from langchain_astradb import AstraDBVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import BedrockEmbeddings
import glob
from pathlib import Path
import json
#from langchain_community.document_loaders import UnstructuredMarkdownLoader
#from unstructured.partition.text import partition_text
#from unstructured.cleaners.core import group_broken_paragraphs

import os
import boto3

token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
keyspace=os.environ['ASTRA_DB_KEYSPACE']

aws_access_key = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
aws_session_key = os.environ["AWS_SESSION_TOKEN"]

bedrock_runtime = boto3.client("bedrock-runtime", "us-west-2")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_runtime)

vstore = AstraDBVectorStore(
    embedding=bedrock_embeddings,
    collection_name="aws_astra_academic",
    api_endpoint=api_endpoint,
    token=token,
    namespace=keyspace,
)

#text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
#    chunk_size = 2000,
#    chunk_overlap  = 50,
#    length_function = len,
#    is_separator_regex = False,
#)

directory = 'captions'
ids = []
llmtexts = []
book = []
section = []
source = []
for filename in glob.glob(directory + '/*.txt'):
    with open(filename) as f:
        file_section = Path(filename).stem
        contents = f.read()
        text = contents.replace('\n\n', '')
        text = text.replace('\n', '')
        llmtexts.append(text)
        ids.append(file_section)
        book.append("10002000123")
        section.append(file_section)
        source.append(filename)


metadatas = [{"book": b, "section": s,"source": t } for b, s, t in zip(book, section, source)]



inserted_ids = vstore.add_texts(texts=llmtexts, metadatas=metadatas, ids=ids)
print(f"\nInserted {len(inserted_ids)} documents.")


#loader = UnstructuredFileLoader("captions/10002000123376.txt")
#docs = loader.load()
#chunks = partition_text(filename="captions/10002000123376.txt", chunking_strategy="basic", paragraph_grouper=group_broken_paragraphs)
#chunks = chunk_elements(elements)
#markdown_path = "captions/10002000123376.txt"
#loader = UnstructuredMarkdownLoader(markdown_path)
#data = loader.load()

#loader = TextLoader("captions/100020001235.txt")
#docs = loader.load()
#texts = text_splitter.split_documents(docs)



#for chunk in chunks[:20]:
#    print(chunk)
#    print("\n")
#print(texts)




