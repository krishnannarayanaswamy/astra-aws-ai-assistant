from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(
    "Sample_notes_6-10.pdf", strategy="hi_res", mode="elements"
)

docs = loader.load()

print (docs[:20])