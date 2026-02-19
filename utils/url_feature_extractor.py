import re
import requests
from urllib.parse import urlparse

def extract_features(url):

    features = []

    parsed = urlparse(url)

    # 1. Having IP Address
    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    features.append(-1 if re.search(ip_pattern, parsed.netloc) else 1)

    # 2. URL Length
    features.append(-1 if len(url) >= 75 else 1)

    # 3. Having @ symbol
    features.append(-1 if "@" in url else 1)

    # 4. Double slash redirect
    features.append(-1 if url.count("//") > 1 else 1)

    # 5. Prefix-Suffix (- in domain)
    features.append(-1 if "-" in parsed.netloc else 1)

    # 6. Subdomain count
    if parsed.netloc.count(".") > 2:
        features.append(-1)
    else:
        features.append(1)

    # 7. HTTPS
    features.append(1 if parsed.scheme == "https" else -1)

    # Fill remaining features with safe defaults (1)
    while len(features) < 30:
        features.append(1)

    return features
