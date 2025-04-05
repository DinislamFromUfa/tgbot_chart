from aiogram import types
from .base import BaseHandler

class CommandHandlers:
    @staticmethod
    async def start(message: types.Message):
        keyboard = BaseHandler.get_keyboard()
        await message.answer(
            "Какой тип графика следует построить?",
            reply_markup=keyboard
        )