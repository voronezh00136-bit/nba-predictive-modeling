"""
NBA box score scraper.

Fetches daily game logs and injury reports from Basketball-Reference
and stores them as raw CSV files ready for cleaning.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.basketball-reference.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NBAPredictor/1.0; +https://github.com/"
        "voronezh00136-bit/nba-predictive-modeling)"
    )
}
REQUEST_DELAY = 3.5  # seconds between requests to respect rate limits


def _get(url: str, timeout: int = 30) -> Optional[BeautifulSoup]:
    """Fetch *url* and return a parsed BeautifulSoup object, or None on error."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    except requests.RequestException as exc:
        logger.error("Request failed for %s: %s", url, exc)
        return None


def scrape_box_scores(season: int = 2024) -> pd.DataFrame:
    """
    Scrape team box scores for every game in *season*.

    Parameters
    ----------
    season:
        NBA season end-year (e.g., 2024 means the 2023-24 season).

    Returns
    -------
    pd.DataFrame
        Raw box-score data with one row per team-game.
    """
    months = [
        "october", "november", "december", "january",
        "february", "march", "april",
    ]
    all_games: list[dict] = []

    for month in months:
        url = f"{BASE_URL}/leagues/NBA_{season}_games-{month}.html"
        logger.info("Scraping schedule: %s", url)
        soup = _get(url)
        if soup is None:
            continue

        table = soup.find("table", id="schedule")
        if table is None:
            logger.warning("No schedule table found for %s %s", month, season)
            continue

        for row in table.find("tbody").find_all("tr"):
            if row.get("class") == ["thead"]:
                continue
            cells = row.find_all(["td", "th"])
            if len(cells) < 7:
                continue

            date_tag = row.find("th")
            if date_tag is None:
                continue

            game: dict = {
                "date": date_tag.get_text(strip=True),
                "away_team": cells[1].get_text(strip=True),
                "away_pts": cells[2].get_text(strip=True),
                "home_team": cells[3].get_text(strip=True),
                "home_pts": cells[4].get_text(strip=True),
            }
            box_link_tag = cells[-1].find("a")
            if box_link_tag:
                game["box_score_url"] = BASE_URL + box_link_tag["href"]
            all_games.append(game)

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_games)
    out_path = RAW_DATA_DIR / f"box_scores_{season}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d game rows to %s", len(df), out_path)
    return df


def scrape_player_game_logs(
    player_slug: str,
    season: int = 2024,
) -> pd.DataFrame:
    """
    Scrape per-game stats for a single player from Basketball-Reference.

    Parameters
    ----------
    player_slug:
        Basketball-Reference player identifier (e.g., ``"jamesle01"``).
    season:
        NBA season end-year.

    Returns
    -------
    pd.DataFrame
        Player game log with points, rebounds, assists, and shooting stats.
    """
    url = (
        f"{BASE_URL}/players/{player_slug[0]}/{player_slug}/"
        f"gamelog/{season}"
    )
    logger.info("Scraping player log: %s", url)
    soup = _get(url)
    if soup is None:
        return pd.DataFrame()

    table = soup.find("table", id="pgl_basic")
    if table is None:
        logger.warning("Game log table not found for %s", player_slug)
        return pd.DataFrame()

    rows = []
    for row in table.find("tbody").find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue
        cells = {td.get("data-stat", ""): td.get_text(strip=True) for td in row.find_all(["td", "th"])}
        cells["player_slug"] = player_slug
        rows.append(cells)

    df = pd.DataFrame(rows)
    out_path = RAW_DATA_DIR / f"player_log_{player_slug}_{season}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows for player %s", len(df), player_slug)
    time.sleep(REQUEST_DELAY)
    return df


def scrape_injury_report(season: int = 2024) -> pd.DataFrame:
    """
    Scrape the current NBA injury report from ESPN.

    Returns
    -------
    pd.DataFrame
        Injury report with player name, team, status, and description.
    """
    url = "https://www.espn.com/nba/injuries"
    logger.info("Scraping injury report from %s", url)
    soup = _get(url)
    if soup is None:
        return pd.DataFrame()

    records = []
    for team_section in soup.find_all("div", class_="Table__league-injuries"):
        team_tag = team_section.find("span", class_="injuries__teamName")
        team_name = team_tag.get_text(strip=True) if team_tag else "Unknown"
        for row in team_section.find_all("tr", class_="Table__TR"):
            cols = row.find_all("td")
            if len(cols) >= 3:
                records.append(
                    {
                        "team": team_name,
                        "player": cols[0].get_text(strip=True),
                        "status": cols[1].get_text(strip=True),
                        "description": cols[2].get_text(strip=True),
                    }
                )

    df = pd.DataFrame(records)
    out_path = RAW_DATA_DIR / f"injuries_{season}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d injury rows to %s", len(df), out_path)
    return df
