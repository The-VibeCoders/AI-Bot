import requests
from bs4 import BeautifulSoup
import traceback
import sys

def get_web_context(query: str, max_results: int = 3) -> str:
    """
    Searches DuckDuckGo and scrapes text from the top results.
    Enhanced with better error handling and logging.
    """
    print(f"\n   [WEB] Searching live internet for: '{query}'")
    
    # Validate input
    if not query or not query.strip():
        print("   [WEB] Empty query, skipping search")
        return ""
    
    query = query.strip()
    if len(query) < 2:
        print("   [WEB] Query too short, skipping search")
        return ""
    
    results = None
    search_error = None
    
    # Try DuckDuckGo first with DDGS library
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        print(f"   [WEB] DDGS returned {len(results) if results else 0} results")
    except ImportError as e:
        print(f"   [WEB] DDGS module not found, trying fallback")
        results = _duckduckgo_fallback(query, max_results)
    except Exception as e:
        print(f"   [WEB] DDGS error: {e}")
        search_error = str(e)
        results = _duckduckgo_fallback(query, max_results)
    
    if not results:
        print(f"   [WEB] ERROR: No search results returned (error: {search_error})")
        return ""
    
    if results and len(results) == 0:
        print(f"   [WEB] ERROR: Empty results list")
        return ""
    
    print(f"   [WEB] Got {len(results)} results, scraping content...")
    
    scraped_text = []
    success_count = 0
    
    for i, res in enumerate(results):
        url = res.get("href") or res.get("url", "")
        title = res.get("title", "No title")
        snippet = res.get("body", "") or res.get("snippet", "")
        
        if not url:
            print(f"   [WEB] Skipping result {i}: no URL")
            continue
        
        try:
            page_content = _scrape_url(url)
            if page_content and len(page_content) > 50:
                scraped_text.append(f"Source: {title} ({url})\nSummary: {snippet}\nDetails: {page_content[:800]}")
                success_count += 1
                print(f"   [WEB] OK Scraped: {title[:30]}...")
            else:
                scraped_text.append(f"Source: {title} ({url})\nSummary: {snippet}")
                print(f"   [WEB] -- Used snippet only: {title[:30]}...")
        except Exception as e:
            print(f"   [WEB] FAIL: Could not scrape {title[:30]}...: {e}")
            scraped_text.append(f"Source: {title} ({url})\nSummary: {snippet}")
    
    if not scraped_text:
        print(f"   [WEB] ERROR: No content scraped from any result")
        return ""
    
    result = "\n\n".join(scraped_text)
    print(f"   [WEB] DONE: Scraped {success_count}/{len(results)} pages")
    return result


def _duckduckgo_fallback(query: str, max_results: int) -> list:
    """Fallback search using requests to DuckDuckGo HTML."""
    try:
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query, "b": max_results}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        resp = requests.post(url, data=data, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        
        for result in soup.select(".result__a")[:max_results]:
            href = result.get("href", "")
            if href:
                results.append({
                    "href": href,
                    "title": result.get_text(),
                    "body": ""
                })
        
        return results
    except Exception as e:
        print(f"   [WEB] Fallback search failed: {e}")
        return []


def _scrape_url(url: str, timeout: tuple = (5, 10)) -> str | None:
    """
    Scrape a URL and extract text content.
    Returns up to 1000 characters of text.
    """
    try:
        # Skip non-HTML URLs
        if any(x in url.lower() for x in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.mp4', '.zip', '.doc']):
            return None
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        
        if response.status_code != 200:
            return None
        
        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # Get text from paragraphs
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return text[:1000] if len(text) > 1000 else text
        
        # Fallback: get all text
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())  # Normalize whitespace
        return text[:1000] if len(text) > 1000 else text
        
    except requests.exceptions.Timeout:
        print(f"   [WEB] Timeout scraping {url[:50]}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"   [WEB] Connection error for {url[:50]}")
        return None
    except Exception as e:
        print(f"   [WEB] Scrape error for {url[:50]}: {e}")
        return None
