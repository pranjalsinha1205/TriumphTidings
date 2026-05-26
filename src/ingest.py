from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

URLS = [
    "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety",
    "https://www.helpguide.org/mental-health/depression/coping-with-depression",
    "https://www.helpguide.org/mental-health/stress/stress-management",
    "https://www.helpguide.org/mental-health/wellbeing/building-better-mental-health",
    "https://www.helpguide.org/relationships/social-connection/loneliness-and-social-isolation",
    "https://www.helpguide.org/mental-health/grief/coping-with-grief-and-loss",
    "https://www.helpguide.org/relationships/communication/conflict-resolution-skills",
    "https://www.helpguide.org/mental-health/depression/helping-someone-with-depression",
    "https://www.helpguide.org/mental-health/wellbeing/laughter-is-the-best-medicine",
    "https://www.helpguide.org/mental-health/depression/depression-in-men",
    "https://www.helpguide.org/mental-health/depression/depression-in-women",
    "https://www.helpguide.org/mental-health/depression/depression-symptoms-and-warning-signs",
    "https://www.helpguide.org/mental-health/depression/depression-treatment",
    "https://www.helpguide.org/mental-health/depression/parents-guide-to-teen-depression",
    "https://www.helpguide.org/mental-health/depression/seasonal-affective-disorder-sad",
    "https://www.helpguide.org/mental-health/depression/teenagers-guide-to-depression",
    "https://www.helpguide.org/mental-health/depression/depression-types-causes-and-risk-factors",
    "https://www.helpguide.org/mental-health/depression/i-feel-depressed",
    "https://www.helpguide.org/mental-health/depression/high-functioning-depression",
    "https://www.helpguide.org/mental-health/depression/sleep-and-depression-connection",
    "https://www.helpguide.org/mental-health/addiction/self-medicating",
    "https://www.helpguide.org/mental-health/suicide-self-harm/are-you-feeling-suicidal",
    "https://www.helpguide.org/mental-health/suicide-self-harm/suicide-prevention-tips",
    "https://www.helpguide.org/mental-health/treatment/cognitive-behavioral-therapy-cbt",
    "https://www.helpguide.org/wellness/career/mental-health-in-the-workplace"
]

def ingest_docs(URLS: list[str]):
    URLS = list(set(URLS))

    loader = WebBaseLoader(URLS)
    docs = loader.load()

    chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = chunker.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="triumph_tidings",
        persist_directory="./chroma_db"
    )

    return vectorstore

if __name__ == "__main__":
    ingest_docs(URLS)