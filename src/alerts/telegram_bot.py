"""
Telegram Bot alert integration.

Sends high-value statistical deviation alerts to a configured Telegram chat.

Configuration (environment variables or ``config/secrets.yaml``):
    TELEGRAM_BOT_TOKEN  — Bot API token obtained from BotFather.
    TELEGRAM_CHAT_ID    — Target chat or channel ID.

Usage example::

    from src.alerts.telegram_bot import TelegramAlerter

    alerter = TelegramAlerter()
    alerter.send_game_alert(
        home_team="Lakers",
        away_team="Celtics",
        home_win_prob=0.62,
        edge=0.08,
    )
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.error import TelegramError
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    logger.warning(
        "python-telegram-bot is not installed. "
        "Install it with: pip install python-telegram-bot"
    )


def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable, falling back to *default*."""
    return os.environ.get(key, default)


class TelegramAlerter:
    """
    Sends NBA prediction alerts to a Telegram chat.

    Parameters
    ----------
    token:
        Telegram Bot API token.  Defaults to the ``TELEGRAM_BOT_TOKEN``
        environment variable.
    chat_id:
        Target Telegram chat ID.  Defaults to the ``TELEGRAM_CHAT_ID``
        environment variable.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self._token = token or _get_env("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or _get_env("TELEGRAM_CHAT_ID")
        self._bot: Optional[object] = None

        if not self._token or not self._chat_id:
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. "
                "Alerts will be logged only."
            )

    def _get_bot(self):
        """Lazily initialise and return the Telegram Bot instance."""
        if not HAS_TELEGRAM:
            raise RuntimeError(
                "python-telegram-bot is not installed. "
                "Run: pip install python-telegram-bot"
            )
        if self._bot is None:
            self._bot = Bot(token=self._token)
        return self._bot

    async def _send(self, text: str) -> bool:
        """
        Internal async helper that dispatches a message via the Bot API.

        Returns *True* on success, *False* on failure.
        """
        if not self._token or not self._chat_id:
            logger.info("[ALERT – log only] %s", text)
            return False
        try:
            bot = self._get_bot()
            await bot.send_message(chat_id=self._chat_id, text=text, parse_mode="Markdown")
            return True
        except TelegramError as exc:
            logger.error("Failed to send Telegram message: %s", exc)
            return False

    def send_message(self, text: str) -> bool:
        """
        Send a raw text message synchronously.

        Falls back to logging when the library or credentials are unavailable.
        """
        import asyncio

        if not self._token or not self._chat_id:
            logger.info("[ALERT – log only] %s", text)
            return False

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._send(text))
        except RuntimeError:
            # No running event loop – create a temporary one
            return asyncio.run(self._send(text))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def send_game_alert(
        self,
        home_team: str,
        away_team: str,
        home_win_prob: float,
        edge: Optional[float] = None,
    ) -> bool:
        """
        Send a game-outcome probability alert.

        Parameters
        ----------
        home_team / away_team:
            Team names.
        home_win_prob:
            Model's predicted probability that the home team wins.
        edge:
            Optional market edge (model_prob − market_implied_prob).
        """
        edge_str = f"\n📈 *Edge vs market:* {edge:+.1%}" if edge is not None else ""
        text = (
            f"🏀 *NBA Game Alert*\n"
            f"*{away_team}* @ *{home_team}*\n"
            f"🏠 Home win probability: *{home_win_prob:.1%}*"
            f"{edge_str}"
        )
        return self.send_message(text)

    def send_prop_alert(
        self,
        player_name: str,
        stat: str,
        line: float,
        model_prob: float,
        edge: float,
        market_odds: Optional[float] = None,
    ) -> bool:
        """
        Send a player prop value-bet alert.

        Parameters
        ----------
        player_name:
            Player's full name.
        stat:
            Stat category (e.g. ``"Points"``).
        line:
            Prop line (e.g. 24.5).
        model_prob:
            Model's estimated "over" probability.
        edge:
            model_prob − market_implied_prob.
        market_odds:
            Market odds (American) for context.
        """
        odds_str = f" (line: {market_odds:+.0f})" if market_odds is not None else ""
        text = (
            f"🔥 *Value Prop Alert*\n"
            f"👤 *{player_name}* — {stat} O{line}{odds_str}\n"
            f"🤖 Model probability: *{model_prob:.1%}*\n"
            f"📈 Edge: *{edge:+.1%}*"
        )
        return self.send_message(text)
