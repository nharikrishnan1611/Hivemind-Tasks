import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time
from datetime import datetime

BASE_URL = "https://quotes.toscrape.com/"
OUTPUT_FILE = "scraped_data.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
logging.basicConfig(
    filename="scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def fetch_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        logging.info(f"Successfully fetched {url}")
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

def parse_quotes(html):
    soup = BeautifulSoup(html, "lxml")
    quotes_data = []

    quotes = soup.find_all("div", class_="quote")

    for quote in quotes:
        text = quote.find("span", class_="text").get_text(strip=True)
        author = quote.find("small", class_="author").get_text(strip=True)
        tags = [tag.get_text(strip=True) for tag in quote.find_all("a", class_="tag")]

        quotes_data.append({
            "quote": text,
            "author": author,
            "tags": ", ".join(tags)
        })

    return quotes_data

def run_pipeline():

    logging.info("Starting Scraping Pipeline")
    all_data = []
    page = 1

    while True:
        url = f"{BASE_URL}page/{page}/"
        print(f"Scraping Page: {page}")

        html = fetch_page(url)

        if html is None:
            break

        data = parse_quotes(html)

        if not data:
            break

        all_data.extend(data)
        page += 1

        time.sleep(1)  

    if all_data:
        df = pd.DataFrame(all_data)
        df.drop_duplicates(inplace=True)
        df["scraped_at"] = datetime.now()

        df.to_csv(OUTPUT_FILE, index=False)

        logging.info("Data saved successfully")
        print("\nScraping Completed Successfully.")
        print(f"Total Records Collected: {len(df)}")

    else:
        print("No data found.")
        logging.warning("No data extracted.")

if __name__ == "__main__":
    run_pipeline()
