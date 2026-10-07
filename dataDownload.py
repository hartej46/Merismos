from datasets import packaged_modules
import os
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from pathlib import Path
from datasets import load_dataset

cache_dir = Path("./dataset/.cache_code")
output_dir = Path("./dataset/code_csv")
cache_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading / Loading dataset...")
ds = load_dataset(
    "flytech/python-codes-25k",
    split="train",
    cache_dir=str(cache_dir)
)

print(f"Loaded {len(ds)} rows. NoW converting it to csv")
ds.to_csv(output_dir / "train.csv")
print("CSV, its done bro")

print("Downloading Tiny Shakespeare dataset")
shakespeare_url = "https://raw.githubusercontent.com/yay1503/GPT-2/refs/heads/main/input.txt"
urllib.request.urlretrieve(shakespeare_url, "./dataset/input.txt")
print("Tiny Shakespeare Downloaded")
