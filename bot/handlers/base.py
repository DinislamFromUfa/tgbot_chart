from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

class BaseHandler:
    FUNCTION_TYPES = {
        "Линейная": "3*x + 2",
        "Квадратичная": "x**2 - 4",
        "Кубическая": "x**3 - 2*x",
        "Рациональная": "(x+1)/(x-2)",
        "Логарифмическая": "log2(x+5)",
        "Тригонометрическая": "sin(x) + cos(2*x)"
    }

    @classmethod
    def get_keyboard(cls):
        return ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text=name)] for name in cls.FUNCTION_TYPES
            ],
            resize_keyboard=True,
            input_field_placeholder="Выберите тип функции"
        )