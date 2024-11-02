import os
import requests
import time
from playwright.sync_api import sync_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed
from github import Github, GithubException
from tqdm import tqdm

from crawl_w_css import fetch_and_embed_css
from screenshot import take_screenshot

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN') 
MAX_RETRIES = 5
BACKOFF_FACTOR = 2  # Exponential backoff multiplier

MAX_WORKERS = 32


gh = Github(GITHUB_TOKEN)

def api_request_with_retry(url, headers):
    retries = 0
    while retries < MAX_RETRIES:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        elif response.status_code == 403:
            # Handle rate limiting
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            sleep_duration = max(reset_time - time.time(), 60) * BACKOFF_FACTOR ** retries  # Wait at least 60 seconds
            print(f"Rate limit reached. Sleeping for {sleep_duration} seconds.")
            time.sleep(sleep_duration)
        elif response.status_code >= 500:
            # Retry on server errors
            retries += 1
            sleep_duration = BACKOFF_FACTOR ** retries
            print(f"Server error {response.status_code}. Retrying in {sleep_duration} seconds...")
            time.sleep(sleep_duration)
        else:
            print(f"Error fetching {url}: {response.status_code} - {response.text}")
            return None
    return None

def get_github_io_sites():
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    github_io_sites = []
    
    # Adjust the date ranges based on how far back you want to search
    date_ranges = [
        "2024-01-01..2024-12-31",
        "2023-01-01..2023-12-31",
        "2022-01-01..2022-12-31",
        "2021-01-01..2021-12-31",
        "2020-01-01..2020-12-31",
        "2019-01-01..2019-12-31",
        "2018-01-01..2018-12-31",
        "2010-01-01..2017-12-31"
    ]
    
    for date_range in date_ranges:
        for page in range(1, 11):  # 10 pages of 100 results each
            url = f"https://api.github.com/search/repositories?q=github.io+in:name+created:{date_range}&page={page}&per_page=100"
            print(f"Fetching url: {url}")
            response = api_request_with_retry(url, headers)
            if response and response.status_code == 200:
                data = response.json()
                repositories = data.get('items', [])
                if not repositories:
                    break
                for repo in repositories:
                    if repo['name'].endswith('.github.io'):
                        github_io_sites.append(repo)
            elif response is None:
                print("Failed to fetch repositories after retries. Exiting.")
                break
    return github_io_sites


def check_license(repo):
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    license_url = repo['url'] + '/license'
    response = api_request_with_retry(license_url, headers)
    if response and response.status_code == 200:
        license_info = response.json().get('license')
        if license_info and license_info.get('spdx_id') in ['MIT', 'Apache-2.0', 'GPL-3.0', 'BSD-3-Clause']:
            return True
    return False

def process_site(repo):
    base_dir = '/juice2/scr2/nlp/pix2code/zyanzhe/Github_Pages'
    
    site_url = f"https://{repo['name']}"
    if check_license(repo):
        # print(f"Processing: {site_url}")
        html_content = fetch_and_embed_css(site_url)
        repo_name = repo['name'].replace(".github.io", "")
        if html_content:
            # print(f"Successfully processed {site_url}")
            with open(f'{base_dir}/{repo_name}.html', 'w') as f:
                f.write(html_content)
            take_screenshot(f'{base_dir}/{repo_name}.html', f'{base_dir}/{repo_name}.png', do_it_again=True)
        else:
            print(f"Failed to process {site_url}")
    else:
        print(f"License check failed for {site_url}, skipping.")
        

def main():
    
    sites = get_github_io_sites()
    print(f"There are a total of {len(sites)} sites")
    # sites = sites[:16]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_site, repo): repo for repo in sites}
        with tqdm(total=len(futures), desc="Processing sites") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()  # Retrieve the result to catch exceptions
                except Exception as e:
                    print(f"An error occurred during processing: {e}")
                finally:
                    pbar.update(1)  # Update the progress bar
    
    # count = 0
    # for repo in sites:
    #     if check_license(repo):
    #         site_url = f"https://{repo['name']}"
    #         print(f"Processing: {site_url}")
    #         html_content = fetch_and_embed_css(site_url)
    #         repo_name = repo['name'].replace(".github.io", "")
    #         if html_content:
    #             print(f"Successfully processed {site_url}")
    #             with open(f'{base_dir}/{repo_name}.html', 'w') as f:
    #                 f.write(html_content)
    #             take_screenshot(f'{base_dir}/{repo_name}.html', f'{base_dir}/{repo_name}.png')
    #             count += 1
    #             if count >= 10:
    #                 break
    #         else:
    #             print(f"Failed to process {site_url}")

if __name__ == '__main__':
    main()