from app.services.source_verification import (
    verify_evidence,
    _extract_relevant_passages,
)

claim = "Regular physical activity can improve cardiovascular health."

evidence = {
    "title": "Exercise: Health benefits",
    "url": "https://www.nhs.uk/live-well/exercise/exercise-health-benefits/",
    "snippet": "Regular physical activity provides important health benefits.",
    "search_context": "physical activity health benefits",
}

# First, retrieve and verify the webpage.
result = verify_evidence(claim, evidence)

print("Page retrieved:", result.get("page_retrieved"))
print("Page text extracted:", result.get("page_content_extracted"))

# Then extract relevant passages.
page_text = result.get("page_text", "")

passages = _extract_relevant_passages(
    claim_text=claim,
    page_text=page_text,
    search_context=evidence["search_context"],
)

print("\nExtracted passages:")

for index, passage in enumerate(passages, start=1):
    print(f"\nPassage {index}")
    print("Relevance score:", passage["relevance_score"])
    print("Text:", passage["text"])
    print("Matched keywords:", passage["matched_keywords"])

import requests
from bs4 import BeautifulSoup

url = "https://www.nhs.uk/live-well/exercise/exercise-health-benefits/"

response = requests.get(
    url,
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=15,
)

soup = BeautifulSoup(response.content, "html.parser")

article = soup.find("article") or soup.find("main")

print("Article found:", article is not None)

if article:
    for element in article.find_all(["h1", "h2", "h3", "p"], limit=20):
        text = element.get_text(" ", strip=True)

        if text:
            print(f"\n[{element.name.upper()}] {text[:300]}")