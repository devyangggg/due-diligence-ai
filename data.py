import yfinance as fin
import pandas as pd
import requests 
import re
from bs4 import BeautifulSoup as bs


def getVitals(ticker:str) -> dict:
    data = fin.download(ticker, period="1y")["Close"][ticker]

    df = pd.DataFrame(data)
    df.columns = ["Price"]
    df["Returns"] = df["Price"].pct_change()*100
    df = df.dropna()
    t_info = fin.Ticker(ticker)
    ticker_apple = t_info.info

    comp_info = {
        "peRatio": ticker_apple['trailingPE'], 
        "sector": ticker_apple['sector'], 
        "marketCap": ticker_apple['marketCap'],
        "priceHist": df['Price'].to_list(),
        "dailyReturns": df['Returns'].to_list(),
        'beta':ticker_apple['beta']
        }
    return comp_info


def _make_ticker_map():
    headers = {"User-Agent": "dev devyang.updates@gmail.com"}
    res = requests.get('https://www.sec.gov/files/company_tickers.json', headers=headers)
    data = res.json()
    return {val['ticker']:val['cik_str'] for val in data.values()}, {val['ticker']:val['title'] for val in data.values()}

ticker_to_cik, ticker_to_name = _make_ticker_map()


def get_latest_10k(ticker:str):

    headers = {"User-Agent": "dev devyang.updates@gmail.com"}
    res = requests.get('https://www.sec.gov/files/company_tickers.json',headers=headers)
    data = res.json()

    ticker_to_cik = {val['ticker']:val['cik_str'] for val in data.values()}
    ticker_to_name = {val['ticker']:val['title'] for val in data.values()}
    
    cik = str(ticker_to_cik[ticker])

    comp_name = str(ticker_to_name[ticker])
    
    if len(cik) < 10:
        dif = 10 - len(cik)
        pad = '0' * dif
        cik = pad + cik
        
    res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",headers=headers)
    data = res.json()

    # python 3.15 safety
    #data['Date'] = pd.to_datetime(data['filings']['recent'][''], format='%Y-%m-%d', errors='coerce')

    df = pd.DataFrame({
        'form' : data['filings']['recent']['form'],
        # 'rep_date' : data['data'],
        'accession_num' : data['filings']['recent']['accessionNumber'],
        'filing_date' : data['filings']['recent']['filingDate']
    })

    df['filing_date'] = pd.to_datetime(df['filing_date'])

    forms = df[df['form'] == '10-K']
    sorted_forms = forms.sort_values(by='filing_date')

    latest_form = sorted_forms.iloc[-1]

    fixed_accession = latest_form['accession_num'].replace("-","")

    new_res = requests.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{fixed_accession}/index.json",headers=headers)

    new_res_data = new_res.json()['directory']['item']
    main_form_url = ""
    for item in new_res_data:
        string = item['name']
    
        if re.search(f"{ticker.lower()}",string) and re.search(".htm",string):
            main_form_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{fixed_accession}/{string}"
            break

            
    form_html = requests.get(main_form_url, headers=headers)
    soup = bs(form_html.content,"html.parser")

    for tag in soup.find_all(["ix:nonfraction", "ix:nonnumeric", "ix:header"]):
        tag.unwrap()

    for header in soup.find_all("ix:header"): #headers dont contain anythng useful
        header.decompose()
        
    text = soup.get_text(separator=" ", strip=True)

    start = text.find("UNITED STATES")

    if start == -1:
        start = text.find(comp_name)

    main_text = text[start:]
    return main_text


def slice_10k_specific_items(text):
    sections = {}
    
    # Step 1: Force the parsing engine to start after the Table of Contents.
    # "PART I" followed closely by "Item 1" marks the start of the actual document body.
    body_start_idx = text.find("PART I")
    if body_start_idx == -1:
        body_start_idx = text.lower().find("part i")
        
    if body_start_idx == -1:
        # Fallback to absolute start if "PART I" header is missing
        body_start_idx = 0
    
    body_text = text[body_start_idx:]
    
    # Step 2: Define the target items and their exact sequential content anchors.
    # Note: We use variations with and without punctuation to increase match reliability.
    targets = [
        {"name": "Item 1", "anchors": ["Item 1. Business", "Item 1.  Business", "Item 1.\xa0\xa0\xa0\xa0Business"]},
        {"name": "Item 1A", "anchors": ["Item 1A. Risk Factors", "Item 1A.  Risk Factors", "Item 1A.\xa0\xa0\xa0\xa0Risk Factors"]},
        {"name": "Item 7", "anchors": ["Item 7. Management’s Discussion", "Item 7. Management's Discussion", "Item 7.\xa0\xa0\xa0\xa0Management’s"]},
        {"name": "Item 8", "anchors": ["Item 8. Financial Statements", "Item 8.  Financial Statements", "Item 8.\xa0\xa0\xa0\xa0Financial"]},
        # We need an end anchor for Item 8 so it doesn't grab the rest of the 10-K (Part IV, Signatures, etc.)
        {"name": "Item 9", "anchors": ["Item 9. Changes in", "Item 9.  Changes", "Item 9.\xa0\xa0\xa0\xa0Changes"]}
    ]
    
    # Step 3: Find the starting character index for each target item in the body text
    for target in targets:
        target["start_idx"] = -1
        for anchor in target["anchors"]:
            idx = body_text.find(anchor)
            if idx != -1:
                target["start_idx"] = idx
                break  # Stop searching once a valid variation is matched

    # Step 4: Slice the text between sequential pairs
    for i in range(len(targets) - 1):  # Loop up to Item 8 (ignoring Item 9's block itself)
        current_item = targets[i]
        next_item = targets[i+1]
        
        start = current_item["start_idx"]
        end = next_item["start_idx"]
        
        # If the current item isn't found, skip it
        if start == -1:
            print(f"Warning: Could not locate content body for {current_item['name']}")
            continue
            
        # If the next item is found, slice up to its beginning. 
        # If it's not found, grab text up to the end of the available document.
        if end != -1 and end > start:
            sections[current_item["name"]] = body_text[start:end].strip()
        else:
            sections[current_item["name"]] = body_text[start:].strip()
            
    return sections
