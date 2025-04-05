from aiogram import Bot

from config import config


class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN.get_secret_value())

    @property
    def instance(self):
        return self.bot