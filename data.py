"""Yahoo Finance vitals and paced, asynchronous SEC filing retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re
import time
from typing import Any
from urllib.parse import quote

from bs4 import BeautifulSoup, NavigableString, Tag
import httpx
import pandas as pd
import yfinance as fin


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: str
    accession: str
    filing_date: str
    url: str


@dataclass(frozen=True)
class Filing:
    ref: FilingRef
    text: str
    sections: dict[str, str]


def getVitals(ticker: str) -> dict:
    data = fin.download(ticker, period="1y")["Close"][ticker]

    df = pd.DataFrame(data)
    df.columns = ["Price"]
    df["Returns"] = df["Price"].pct_change() * 100
    df = df.dropna()
    t_info = fin.Ticker(ticker)
    ticker_apple = t_info.info

    comp_info = {
        "peRatio": ticker_apple["trailingPE"],
        "sector": ticker_apple["sector"],
        "marketCap": ticker_apple["marketCap"],
        "priceHist": df["Price"].to_list(),
        "dailyReturns": df["Returns"].to_list(),
        "beta": ticker_apple["beta"],
    }
    return comp_info


_CIK_MAP_TTL = 60 * 60 * 12
_CIK_MAP: dict[str, str] = {}
_CIK_MAP_EXPIRES = 0.0
_CIK_MAP_LOCK = asyncio.Lock()
_SEC_PACE_LOCK = asyncio.Lock()
_SEC_NEXT_REQUEST = 0.0
_SEC_INTERVAL = 0.35
_MAX_RETRIES = 3


class SECClient:
    """Use a lifecycle-managed HTTP client with SEC identity and shared pacing.

    The caller supplies HTTP timeouts and owns closing the underlying client.
    """

    def __init__(self, http: httpx.AsyncClient, user_agent: str) -> None:
        self._http = http
        self.user_agent = user_agent

    def _headers(self) -> dict[str, str]:
        if (not isinstance(self.user_agent, str)
                or not re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", self.user_agent)
                or "\r" in self.user_agent or "\n" in self.user_agent):
            raise ValueError("SEC User-Agent must identify the application and contain a contact email")
        return {"User-Agent": self.user_agent}

    async def _get(self, url: str) -> httpx.Response:
        global _SEC_NEXT_REQUEST
        headers = self._headers()
        for attempt in range(_MAX_RETRIES + 1):
            async with _SEC_PACE_LOCK:
                now = time.monotonic()
                await asyncio.sleep(max(0.0, _SEC_NEXT_REQUEST - now))
                _SEC_NEXT_REQUEST = time.monotonic() + _SEC_INTERVAL
            try:
                response = await self._http.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(min(2 ** attempt * 0.5, 8))
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < _MAX_RETRIES:
                    delay = _retry_delay(response.headers.get("Retry-After"), attempt)
                    await asyncio.sleep(delay)
                    continue
            response.raise_for_status()
            return response
        raise RuntimeError("SEC retry budget exhausted")

    async def _json(self, url: str) -> dict[str, Any]:
        value = (await self._get(url)).json()
        if not isinstance(value, dict):
            raise ValueError(f"SEC returned unexpected JSON object: {url}")
        return value

    async def resolve_filing(self, ticker: str) -> FilingRef:
        ticker = ticker.strip().upper()
        cik = await self._cik_for(ticker)
        submissions = await self._json(f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json")
        try:
            recent = submissions["filings"]["recent"]
            records = zip(recent["form"], recent["filingDate"], recent["accessionNumber"],
                          recent.get("primaryDocument", [""] * len(recent["form"])), strict=True)
            candidates = [(date, accession, primary) for form, date, accession, primary in records
                          if form == "10-K"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed SEC submissions for {ticker}") from exc
        if not candidates:
            raise ValueError(f"No recent 10-K filing found for {ticker}")
        date, accession, primary = max(candidates, key=lambda row: (row[0], row[1]))
        accession_dir = accession.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_dir}/"
        index = await self._json(base + "index.json")
        try:
            entries = index["directory"]["item"]
            names = [entry["name"] for entry in entries if isinstance(entry, dict)]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed SEC filing index for {accession}") from exc
        document = _choose_document(names, primary)
        return FilingRef(ticker=ticker, cik=cik.zfill(10), accession=accession,
                         filing_date=date, url=base + quote(document))

    async def _cik_for(self, ticker: str) -> str:
        global _CIK_MAP, _CIK_MAP_EXPIRES
        if ticker in _CIK_MAP and time.monotonic() < _CIK_MAP_EXPIRES:
            return _CIK_MAP[ticker]
        async with _CIK_MAP_LOCK:
            if time.monotonic() >= _CIK_MAP_EXPIRES:
                entries = await self._json("https://www.sec.gov/files/company_tickers.json")
                try:
                    mapping = {entry["ticker"].upper(): str(entry["cik_str"])
                               for entry in entries.values()}
                except (KeyError, TypeError, AttributeError) as exc:
                    raise ValueError("Malformed SEC company ticker directory") from exc
                _CIK_MAP, _CIK_MAP_EXPIRES = mapping, time.monotonic() + _CIK_MAP_TTL
            try:
                return _CIK_MAP[ticker]
            except KeyError as exc:
                raise ValueError(f"Unknown SEC ticker: {ticker}") from exc

    async def fetch_filing(self, ref: FilingRef) -> Filing:
        response = await self._get(ref.url)
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in list(soup.find_all(True)):
            if tag.attrs is None:
                continue
            if tag.name in {"script", "style", "noscript", "ix:header", "ix:hidden"}:
                tag.decompose()
            elif tag.get("hidden") is not None or re.search(
                r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", tag.get("style", ""), re.I
            ):
                tag.decompose()
        for tag in list(soup.find_all(["ix:nonfraction", "ix:nonnumeric", "ix:continuation"])):
            tag.unwrap()
        text = _filing_text(soup)
        return Filing(ref=ref, text=text, sections=slice_10k_specific_items(text))


def _retry_delay(header: str | None, attempt: int) -> float:
    if header:
        try:
            seconds = float(header)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                seconds = 0.0
        return max(0.0, min(seconds, 30.0)) if math.isfinite(seconds) else 0.0
    return min(2 ** attempt * 0.5, 8.0)


def _choose_document(names: list[str], primary: str | None) -> str:
    html = [name for name in names if isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+\.html?", name, re.I)]
    if isinstance(primary, str):
        for name in html:
            if name.casefold() == primary.casefold():
                return name
    eligible = [name for name in html if not re.search(
        r"(?:^|[-_.])(?:ex(?:hibit)?[-_.]?\d+|index|cover|graphic|xbrl)(?:[-_.]|$)", name, re.I
    ) and not re.match(r"^(?:cal|def|lab|pre)[-_.]", name, re.I)]
    form_names = [name for name in eligible if re.search(r"(?:10[-_]?k|form[-_]?10[-_]?k)", name, re.I)]
    if len(form_names) == 1:
        return form_names[0]
    if len(eligible) == 1:
        return eligible[0]
    raise ValueError("SEC index has no unambiguous primary HTML 10-K document")


_BLOCKS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
           "li", "table", "tr", "blockquote", "ul", "ol"}


def _filing_text(soup: BeautifulSoup) -> str:
    """Retain human-readable block boundaries, without inserting spaces inside inline text."""
    pieces: list[str] = []

    def visit(node: Tag | NavigableString) -> None:
        if isinstance(node, NavigableString):
            pieces.append(str(node).replace("\xa0", " "))
        elif isinstance(node, Tag):
            if node.name in _BLOCKS:
                pieces.append("\n\n")
            elif node.name == "br":
                pieces.append("\n")
            for child in node.children:
                visit(child)
            if node.name in _BLOCKS:
                pieces.append("\n\n")
            elif node.name in {"td", "th"}:
                pieces.append(" ")

    visit(soup.body or soup)
    text = "".join(pieces)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]*\n[ \t]*", "\n", text)).strip()


# Bump when section extraction changes; cached text is reparsed once per version.
PARSER_VERSION = "sec-headings-v2"

_HEADING = re.compile(r"(?im)^[ \t]*item[ \t]+(1a|1b|1c|7a|9a|10|11|12|13|14|15|1|2|3|4|5|6|7|8|9)\b[ \t]*[.\u2014:\-]?[ \t]*([^\n]{0,140})$")
_REQUIRED = {"1": "Business", "1a": "Risk Factors", "7": "Management", "8": "Financial"}
_NEXT_ITEM = {
    "1": {"1a"},
    "1a": {"1b", "1c", "2"},
    "7": {"7a", "8"},
    "8": {"9", "9a", "10"},
}
_ORDER = ("1", "1a", "7", "8")


def slice_10k_specific_items(text: str) -> dict[str, str]:
    """Extract actual item bodies, rejecting TOC entries and absent end boundaries."""
    matches: list[tuple[str, int, int]] = []
    for match in _HEADING.finditer(text):
        number, title = match.group(1).lower(), match.group(2).strip()
        # TOC rows carry page numbers; heading titles may have a trailing page number.
        if re.search(r"\s\d{1,4}\s*$", title):
            continue
        if number in _REQUIRED and title and not title.lower().startswith(_REQUIRED[number].lower()):
            if number != "7" or "discussion" not in title.lower():
                continue
        matches.append((number, match.start(), match.end()))

    selected: dict[str, tuple[int, int, int]] = {}
    for pos, (number, start, end) in enumerate(matches):
        if number not in _REQUIRED:
            continue
        # Running page headers repeat the current item number. The boundary is
        # the next distinct SEC item, not the next repeated page heading.
        next_start = next(
            (candidate_start for candidate_number, candidate_start, _ in matches[pos + 1:]
             if candidate_number in _NEXT_ITEM[number]),
            None,
        )
        if next_start is None or next_start <= end:
            continue
        length = len(text[end:next_start].strip())
        if number not in selected or length > selected[number][2]:
            selected[number] = (end, next_start, length)

    missing = [f"Item {number.upper()}" for number in _ORDER if number not in selected]
    if missing:
        raise ValueError("Missing 10-K item or following boundary: " + ", ".join(missing))
    if any(selected[a][0] >= selected[b][0] for a, b in zip(_ORDER, _ORDER[1:])):
        raise ValueError("10-K items occur out of order")
    sections = {f"Item {number.upper()}": text[selected[number][0]:selected[number][1]].strip()
                for number in _ORDER}
    if any(not section for section in sections.values()):
        raise ValueError("One or more 10-K items have empty bodies")
    return sections
