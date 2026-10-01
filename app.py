import os
import streamlit as st
import json
import urllib.request
from dotenv import load_dotenv
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from sec_vectorstore import query_sec_filing
from rapidfuzz import process, fuzz

def extract_text(response) -> str:
    """Extracts raw string cleanly from Gemini content blocks or dictionaries."""
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    # If it is a list of blocks like [{'type': 'text', 'text': '...'}]
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif hasattr(block, "text"):
                text_parts.append(block.text)
            else:
                text_parts.append(str(block))
        return "\n".join(text_parts)

    # If it is already a dictionary
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])

    return str(content)

# ==========================================
# DYNAMIC SEC TICKER REGISTRY & VALIDATION
# ==========================================
@st.cache_data(ttl=86400) # Cache for 24 hours so it only downloads once
def get_sec_ticker_directory() -> dict:
    """
    Pulls the official, live SEC EDGAR company directory (~10,000+ public companies).
    Returns a dictionary of: { 'TICKER': {'title': 'Company Name', 'cik': '12345'} }
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "BlueprintAI analyst@blueprintai.com"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            # Convert SEC format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
            ticker_map = {}
            for item in data.values():
                t = str(item["ticker"]).upper().strip()
                ticker_map[t] = {
                    "title": item["title"],
                    "cik": str(item["cik_str"])
                }
            return ticker_map
    except Exception as e:
        print(f"Warning: Could not fetch SEC registry ({e}). Falling back to algorithmic checks.")
        return {}


def validate_and_suggest_ticker(raw_input: str) -> tuple[bool, str, str]:
    """
    Validates any user input against the official SEC directory of ~10,000 active tickers.
    If invalid, suggests the closest real ticker using fuzzy matching + Gemini.
    """
    user_symbol = raw_input.strip().upper()
    directory = get_sec_ticker_directory()

    # 1. Exact Match against the official SEC directory
    if user_symbol in directory:
        company_name = directory[user_symbol]["title"]
        return True, user_symbol, company_name

    # 2. Check if the user entered a company name instead (e.g. "Palantir" -> PLTR)
    all_tickers = list(directory.keys())
    all_names = [info["title"] for info in directory.values()]

    # First check fuzzy matching on tickers (e.g., APPL -> AAPL)
    best_ticker_match, ticker_score, _ = process.extractOne(user_symbol, all_tickers, scorer=fuzz.ratio)
    if ticker_score >= 80:
        company_name = directory[best_ticker_match]["title"]
        msg = f"'{user_symbol}' is not a recognized ticker. Did you mean **{best_ticker_match}** ({company_name})?"
        return False, best_ticker_match, msg

    # Next check fuzzy matching on company name (e.g., "Microsoft" -> MSFT)
    best_name_match, name_score, match_idx = process.extractOne(user_symbol, all_names, scorer=fuzz.token_sort_ratio)
    if name_score >= 70:
        matched_ticker = all_tickers[match_idx]
        msg = f"'{raw_input}' looks like a company name. Did you mean ticker **{matched_ticker}** ({best_name_match})?"
        return False, matched_ticker, msg

    # 3. LLM Fallback (handles slang, edge cases, or corporate rebrandings)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an SEC Market Auditor. Identify the primary US exchange ticker for the user's input.
Return a JSON object with:
- "is_valid": true if this input is directly a valid stock ticker, else false.
- "suggested_ticker": standard ticker symbol (e.g., 'PLTR' for 'Palantir'), or null if unknown.
- "company_name": company title or null.
Return ONLY raw JSON."""),
        ("human", "Input: {input}")
    ])
    
    try:
        res = (prompt | llm).invoke({"input": user_symbol})
        raw_json = extract_text(res).strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[-1].rsplit("
