import serpapi
import json
import logging
import os
from dataclasses import dataclass
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scraper")

API_KEY = os.environ["SERPAPI_API_KEY"]
client = serpapi.Client(api_key=API_KEY)

@dataclass
class ScrapedImage:
    url: str
    title: str
    source: str
    thumbnail: str | None = None


def get_result(
    search_query: str,
    engine: str,
    language: str,
    max_result: int = 20
)->list[ScrapedImage]:
    results = client.search({
    "engine": engine,
    "q": search_query,
    "hl": language
    })
    if engine == "google_shopping":
        scraped_results = results.get("shopping_results", [])
        img_param = "thumbnail"
    else: 
        scraped_results = results.get("images_results", [])
        img_param = "original"
    if not scraped_results: 
        log.warning(f"No image results for query: {search_query!r}")
        return []

    images = []
    for item in scraped_results[:max_result]:
        img_url = item.get(img_param)
        if not img_url:
            continue
        images.append(ScrapedImage(
            url=img_url,
            title=item.get("title", ""),
            source=item.get("source", ""),
            thumbnail=item.get("thumbnail"),

        ))
    log.info(f"Found {len(images)} shopping images for {search_query!r}")
    return images

def download_images(images: list[ScrapedImage], out_dir: str) -> list[str]:
    """Downloads image URLs to disk, skipping any that fail. Returns local paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        try:
            resp = requests.get(img.url, timeout=10)
            resp.raise_for_status()
            path = os.path.join(out_dir, f"candidate_{i:02d}.jpg")
            with open(path, "wb") as f:
                f.write(resp.content)
            paths.append(path)
        except Exception as e:
            log.warning(f"Failed to download {img.url}: {e}")
            continue
    log.info(f"Downloaded {len(paths)}/{len(images)} images to {out_dir}")
    return paths

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Web Scraping Process via SerpAPI")
    parser.add_argument("--search_query", type=str, default="coffee")
    parser.add_argument("--engine", type=str, default="google_images")
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    images = get_result(args.search_query, args.engine, args.language)
    if images:
        download_images(images, "./scraped_images")


if __name__ == "__main__":
    main()