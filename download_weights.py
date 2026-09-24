import requests
import os
import re
from urllib.parse import unquote

def get_direct_download(link):
    if "?" in link:
        if "download=1" not in link:
            return link + "&download=1"

    else:
        return link + "&download=1"

    return link

def download_files(links, output_directory="weights"):
    os.makedirs(output_directory, exist_ok=True)

    for index, link in enumerate(links, start=1):
        link = link.strip()
        link = get_direct_download(link)

        try:
            response = requests.get(link, stream=True, allow_redirects=True)
            response.raise_for_status()

            filename = None
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                # Updated regex: Stops safely at semicolons, quotes, or spaces
                match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^;\'"\s]+)["\']?', content_disposition, re.IGNORECASE)
                if match:
                    raw_name = match.group(1)
                    if "UTF-8''" in raw_name:
                        raw_name = raw_name.split("UTF-8''")[-1]
                    filename = unquote(raw_name)

                # if filenames:
                #     filename = unquote(filenames[0])

            file_path = os.path.join(output_directory, filename)

            print(f"Downloading file: {filename}")

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Saved file to: {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading link: {e}")


if __name__ == "__main__":
    weights = [
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQB82ajObABeQorLxHKlbTNpAUTpPLDCUAGUNgtm0Q4Hf44?e=tNDlsX",
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQA1nZoGcW-iQLGIy2He0jCfAX82gz3JIfN4kxYmiCIaSUw?e=adcivK",
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQBbqRp1CtCBRobPyu58-0ifAR3yzX8mfAp2swhlocCMHak?e=0L2WOe",
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQCDxAEbfygpR4JrDDDYd_05ATVLFms3w4dGuZAUHu_Lboc?e=5QywPS",
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQDLicvmRbnsS6ZGgwNtEZBrAZYCzvcxhBIHCs40P2GrVQY?e=CwNjpF",
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQBlI-HYjNCIT5L0BlY6Fek_Aay_URMtyxni-3WenDxEx3U?e=yOUSh0",
    ]

    download_files(weights, output_directory="weights")