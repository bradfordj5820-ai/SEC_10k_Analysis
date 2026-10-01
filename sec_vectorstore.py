import os
import re
from glob import glob
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

SEC_COMPANY_NAME = "BlueprintAI"
SEC_EMAIL = "analyst@blueprintai.com"
CACHE_DIR = "./sec_filings_cache"

_DOCUMENT_STORE = {}

def _is_xbrl_or_noise(line: str) -> bool:
    noise_patterns = [
        r'^\s*["\']?(?:terseLabel|label|documentation|role|auth_ref|xbrltype|nsuri)["\']?\s*:',
        r'^\s*["\']?[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+["\']?\s*:',
        r'^R\d+\.htm\b',
        r'^\d+\s*-\s*Disclosure\s*-',
        r'http://(?:www\.)?[\w\.-]+/(?:role|taxonomy|dei|us-gaap)',
        r'^[\[\{\]\}",\s]+$',
        r'^(?:true|false)$',
    ]
    for pattern in noise_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False

def get_clean_filing_chunks(ticker: str) -> list[Document]:
    ticker = ticker.upper()
    if ticker in _DOCUMENT_STORE:
        return _DOCUMENT_STORE[ticker]

    print(f"-> Fetching Form 10-K for {ticker} from SEC EDGAR...")
    dl = Downloader(SEC_COMPANY_NAME, SEC_EMAIL, CACHE_DIR)
    dl.get("10-K", ticker, limit=1)

    htm_files = glob(os.path.join(CACHE_DIR, "sec-edgar-filings", ticker, "10-K", "*", "*.htm*"))
    txt_files = glob(os.path.join(CACHE_DIR, "sec-edgar-filings", ticker, "10-K", "*", "*.txt"))
    files = htm_files if htm_files else txt_files

    if not files:
        raise FileNotFoundError(f"Could not download or find 10-K filing for {ticker}.")

    with open(files[0], "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    tags_to_remove = ["script", "style", "head", "noscript", "xbrl", "ix:header", "ix:hidden"]
    for tag in soup(tags_to_remove):
        tag.extract()

    text = soup.get_text(separator="\n")
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [line for line in raw_lines if not _is_xbrl_or_noise(line)]
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for line in lines:
        current_chunk.append(line)
        current_len += len(line)
        if current_len >= 1500:
            content = "\n".join(current_chunk)
            chunks.append(Document(page_content=content, metadata={"source": f"{ticker} 10-K"}))
            current_chunk = current_chunk[-4:]
            current_len = sum(len(x) for x in current_chunk)

    if current_chunk:
        chunks.append(Document(page_content="\n".join(current_chunk), metadata={"source": f"{ticker} 10-K"}))

    print(f"-> Parsed {len(chunks)} clean financial chunks for {ticker}.")
    _DOCUMENT_STORE[ticker] = chunks
    return chunks

def query_sec_filing(ticker: str, query: str, k: int = 5) -> list[Document]:
    chunks = get_clean_filing_chunks(ticker)
    corpus_tokens = [re.findall(r"\w+", doc.page_content.lower()) for doc in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = re.findall(r"\w+", query.lower())
    top_docs = bm25.get_top_n(query_tokens, chunks, n=k)
    return top_docs
