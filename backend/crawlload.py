from langchain.docstore.document import Document
from langchain_community.utilities import ApifyWrapper
from langchain_community.embeddings import BedrockEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ApifyDatasetLoader
import os
import boto3

token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
apify_api_key=os.environ["APIFY_API_TOKEN"]

aws_access_key = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
aws_session_key = os.environ["AWS_SESSION_TOKEN"]

bedrock_runtime = boto3.client("bedrock-runtime", "us-west-2")
bedrock_embeddings = BedrockEmbeddings(model_id="cohere.embed-multilingual-v3",
                                       client=bedrock_runtime)

vstore = AstraDBVectorStore(
    embedding=bedrock_embeddings,
    collection_name="sinarmas_chatbot",
    api_endpoint=api_endpoint,
    token=token,
)
apify = ApifyWrapper()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2000,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)

#loader = apify.call_actor(
#   actor_id="apify/website-content-crawler",
#    run_input={"startUrls": [{"url": "https://www.sinarmasland.com/"}]},
##   dataset_mapping_function=lambda item: Document(
#        page_content=item["text"] or "", metadata={"source": item["url"]}
#    ),
#)

loader = ApifyDatasetLoader(
    dataset_id="siy1QOoei7kRq9Srt",
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
    ),
)

docs = loader.load()

texts = text_splitter.split_documents(docs)

#texts = text_splitter.create_documents([docs])
print(texts[0])
print(texts[1])

inserted_ids = vstore.add_documents(texts)
print(f"\nInserted {len(inserted_ids)} documents.")
