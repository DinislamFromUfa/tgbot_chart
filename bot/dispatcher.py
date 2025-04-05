from aiogram import Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

class BotDispatcher:
    def __init__(self):
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)