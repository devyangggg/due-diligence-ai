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
        "priceHist": df['Price'],
        "dailyReturns": df['Returns'],
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
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')

    df = pd.DataFrame({
        'form' : data['filings']['recent']['form'],
        'rep_date' : data['data'],
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

