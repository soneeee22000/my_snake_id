import os
import csv
import time
import json
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

INAT_API_URL = "https://api.inaturalist.org/v1/observations"


# ======================================================
# Helper: safe folder name
# ======================================================
def safe_name(name: str):
    return name.replace(" ", "_").replace("-", "_")


# ======================================================
# Single image download (parallel-safe)
# ======================================================
def download_image(url, filepath):
    try:
        if os.path.exists(filepath):
            return True
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception:
        return False


# ======================================================
# Download all images for a species (fast + safe)
# ======================================================
def download_species_images(species_name: str, out_dir: str, max_photos=200):
    folder_name = safe_name(species_name)
    save_dir = os.path.join(out_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🔍 Searching iNaturalist for: {species_name}")

    page = 1
    downloaded = 0
    species_encoded = quote(species_name)

    # Thread pool for fast image downloading
    executor = ThreadPoolExecutor(max_workers=10)

    while downloaded < max_photos:
        params = f"?taxon_name={species_encoded}&page={page}&per_page=100&photos=true"
        url = INAT_API_URL + params

        try:
            data = urllib.request.urlopen(url).read()
            data = json.loads(data)
        except (HTTPError, URLError):
            print(f"⚠️ Failed to fetch page {page} for {species_name}")
            break

        results = data.get("results", [])
        if not results:
            break

        futures = []

        for obs in results:
            photos = obs.get("photos", [])
            for p in photos:
                img_url = p.get("url")
                if not img_url:
                    continue

                img_url = img_url.replace("square", "original")

                filename = f"{folder_name}_{downloaded+1}.jpg"
                filepath = os.path.join(save_dir, filename)

                futures.append(executor.submit(download_image, img_url, filepath))
                downloaded += 1

                if downloaded >= max_photos:
                    break
            if downloaded >= max_photos:
                break

        # Wait for this batch of downloads to complete
        for _ in as_completed(futures):
            pass

        page += 1

    executor.shutdown(wait=True)
    return downloaded


# ======================================================
# Main
# ======================================================
def main():
    species_csv = "species_list.csv"
    output_dir = "dataset/images"

    os.makedirs(output_dir, exist_ok=True)

    missing_species = []
    download_report = []

    with open(species_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            species = row[0].strip()
            count = download_species_images(species, output_dir)

            if count == 0:
                print(f"❌ No images found for {species}")
                missing_species.append(species)
            else:
                print(f"✅ Downloaded {count} images for {species}")

            download_report.append([species, count])

    with open("missing_species.txt", "w") as f:
        for s in missing_species:
            f.write(s + "\n")

    with open("download_report.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Species", "DownloadedImages"])
        writer.writerows(download_report)

    print("\n📄 Finished downloading!")
    print(" - missing_species.txt created")
    print(" - download_report.csv created")


if __name__ == "__main__":
    main()
