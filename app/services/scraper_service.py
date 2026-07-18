import requests
from bs4 import BeautifulSoup

def get_web_context(query: str, max_results: int = 3) -> str:
    print(f"\n   [WEB] Searching live internet for: '{query}'")
    if not query or len(query.strip()) < 2: return ""
    
    query = query.strip()
    results = None
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
    except Exception:
        results = _duckduckgo_fallback(query, max_results)
    
    if not results: return ""
    scraped_text = []
    
    for i, res in enumerate(results):
        url = res.get("href") or res.get("url", "")
        title = res.get("title", "No title")
        snippet = res.get("body", "") or res.get("snippet", "")
        if not url: continue
        
        try:
            page_content = _scrape_url(url)
            if page_content and len(page_content) > 50:
                scraped_text.append(f"Source: {title} ({url})\nSummary: {snippet}\nDetails: {page_content[:800]}")
            else:
                scraped_text.append(f"Source: {title} ({url})\nSummary: {snippet}")
        except Exception:
            scraped_text.append(f"Source: {title} ({url})\nSummary: {snippet}")
            
    return "\n\n".join(scraped_text) if scraped_text else ""

def _duckduckgo_fallback(query: str, max_results: int) -> list:
    try:
        resp = requests.post("https://html.duckduckgo.com/html/", data={"q": query, "b": max_results}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code != 200: return []
        soup = BeautifulSoup(resp.text, "html.parser")
        return [{"href": r.get("href", ""), "title": r.get_text(), "body": ""} for r in soup.select(".result__a")[:max_results] if r.get("href")]
    except Exception: return []

def _scrape_url(url: str, timeout: tuple = (5, 10)) -> str | None:
    try:
        if any(x in url.lower() for x in ['.pdf', '.png', '.jpg']): return None
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""): return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return text[:1000] if len(text) > 1000 else text
        return soup.get_text(separator=" ", strip=True)[:1000]
    except Exception: return None