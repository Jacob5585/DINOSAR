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
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQCDxAEbfygpR4JrDDDYd_05ATVLFms3w4dGuZAUHu_Lboc?e=LuYHdi", # deit_small_patch16_224.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQBbqRp1CtCBRobPyu58-0ifAR3yzX8mfAp2swhlocCMHak?e=xznqBS", # swin_tiny_patch4_window7_224.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQB82ajObABeQorLxHKlbTNpAUTpPLDCUAGUNgtm0Q4Hf44?e=zRVcCC", # checkpoint_reg_vit_lora.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQA1nZoGcW-iQLGIy2He0jCfAX82gz3JIfN4kxYmiCIaSUw?e=XxCk4Y", # checkpoint_swin_lora.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQBlI-HYjNCIT5L0BlY6Fek_Aay_URMtyxni-3WenDxEx3U?e=51qITq", # checkpoint_reg_vit_scratch.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQDvRix858vjT4z7gxLjFY9lAYQ88MmBcHieookslFDKy_o?e=us2SD5", # checkpoint_swin_scratch.pth
        # "", # checkpoint_reg_vit_lora_havy.pth
        # "", # checkpoint_swin_lora_heavy.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQAKvKol4b_HTKq5ZC1HKBQKAY5Qm6jCElFO629bljG-UCQ?e=zVglpC", # vit_small_lora_fasterrcnn_head_sardet100k.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQCkelKP1-T0SYgavyTYw9jdAdINoZGGVXwp5MWHmunNeHI?e=P8tMso", # swin_tiny_lora_fasterrcnn_head_sardet100k.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQDDUiuTBRDTRpqFwnsLAQvOAesR1sDWKcKawvhQUaW_hi0?e=eL5gu4", # vit_small_scratch_fasterrcnn_head_sardet100k.pth
        "https://sunypoly-my.sharepoint.com/:u:/g/personal/cormiej_sunypoly_edu/IQAZZldNH_h7QZ34eXI6P_AZAcBdkISbXhEFqfXJ7tny4ss?e=ULLw8A", # swin_tiny_scratch_fasterrcnn_head_sardet100k.pth
        # "", # vit_small_lora_heavy_fasterrcnn_head_sardet100k.pth
        # "", # swin_tiny_lora_heavy_fasterrcnn_head_sardet100k.pth
    ]

    download_files(weights, output_directory="weights")

    print("Download Complete")