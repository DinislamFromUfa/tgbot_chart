from aiogram import Router, F, types
from aiogram.filters import Command
from aiogram.types import Message, BufferedInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import numpy as np
import io
import re
import logging
from .base import BaseHandler
import matplotlib.pyplot as plt
import ast
import operator

router = Router()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Безопасный список операций
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,  # Унарный минус
    ast.UAdd: operator.pos  # Унарный плюс
}

# Безопасный список функций
SAFE_FUNCS = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'log': np.log,
    'log2': np.log2,
    'log10': np.log10,
    'sqrt': np.sqrt,
    'exp': np.exp,
    'pi': np.pi,
    'e': np.e
}

# Ограничения на область определения для функций
DOMAIN_RESTRICTIONS = {
    'log': lambda x: x > 0,
    'log2': lambda x: x > 0,
    'log10': lambda x: x > 0,
    'sqrt': lambda x: x >= 0,
    # Можно добавить другие функции с ограничениями, например, 'asin', 'acos'
}


class FunctionState(StatesGroup):
    waiting_formula = State()


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Выберите тип функции:",
        reply_markup=BaseHandler.get_keyboard()
    )


@router.message(F.text.in_(BaseHandler.FUNCTION_TYPES))
async def select_function(message: Message, state: FSMContext):
    await state.set_state(FunctionState.waiting_formula)
    await state.update_data(func_type=message.text)
    example = BaseHandler.FUNCTION_TYPES[message.text]
    await message.answer(
        f"Введите формулу (пример: {example})\n"
        "Можно писать сокращения: 3x → 3*x, 2sin(x) → 2*sin(x)\n"
        "Обратите внимание: логарифмы определены только для x > 0, квадратный корень — для x >= 0.",
        reply_markup=types.ReplyKeyboardRemove()
    )


@router.message(FunctionState.waiting_formula)
async def process_formula(message: Message, state: FSMContext):
    try:
        data = await state.get_data()
        func_type = data["func_type"]

        await state.clear()


        image_bytes, error_message = create_plot(func_type, message.text)

        if image_bytes:

            photo = BufferedInputFile(image_bytes, filename="plot.png")

            await message.answer_photo(
                photo=photo,
                caption=f"График функции: {message.text}"
            )
        else:
            await message.answer(
                f"Ошибка: {error_message}\nПопробуйте ещё раз.",
                reply_markup=BaseHandler.get_keyboard()
            )
            return  # Выходим из функции

        # Возвращаем клавиатуру
        await message.answer(
            "Хотите построить ещё один график?",
            reply_markup=BaseHandler.get_keyboard()
        )

    except Exception as e:
        await message.answer(
            f"Непредвиденная ошибка: {str(e)}\nПопробуйте ещё раз.",
            reply_markup=BaseHandler.get_keyboard()
        )


def get_domain_restriction(formula):
    """Определяет, есть ли в формуле функции с ограниченной областью определения"""
    for func_name, restriction in DOMAIN_RESTRICTIONS.items():
        if func_name in formula:
            return restriction
    return None


def create_plot(func_type, formula):
    """Создание графика с использованием Matplotlib и ast"""
    try:
        # Преобразуем формулу
        expr = formula.replace("^", "**")
        expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)  # Добавляем умножение

        # Компилируем формулу в AST
        tree = ast.parse(expr, mode='eval')

        # Функция для безопасного вычисления
        def eval_expr(node, x_val):
            if isinstance(node, ast.Num):  # Число
                return node.n
            elif isinstance(node, ast.Name):  # Переменная или константа
                if node.id == 'x':
                    return x_val
                elif node.id in SAFE_FUNCS:
                    return SAFE_FUNCS[node.id]  # Возвращаем значение или функцию
                else:
                    raise ValueError(f"Недопустимая переменная или функция: {node.id}")
            elif isinstance(node, ast.BinOp):  # Бинарная операция
                op = SAFE_OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Недопустимая операция: {node.op}")
                left = eval_expr(node.left, x_val)
                right = eval_expr(node.right, x_val)

                # Если один из операндов — функция numpy, применяем её к аргументу
                if callable(left):
                    left = left(x_val)
                if callable(right):
                    right = right(x_val)

                return op(left, right)
            elif isinstance(node, ast.UnaryOp):  # Унарная операция
                op = SAFE_OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Недопустимая унарная операция: {node.op}")
                operand = eval_expr(node.operand, x_val)
                if callable(operand):
                    operand = operand(x_val)
                return op(operand)
            elif isinstance(node, ast.Call):  # Вызов функции (например, sin(x), log2(x))
                func = eval_expr(node.func, x_val)
                if not callable(func):
                    raise ValueError(f"Объект {func} не является функцией")
                args = [eval_expr(arg, x_val) for arg in node.args]
                return func(*args)  # Вызываем функцию с аргументами
            else:
                raise ValueError(f"Недопустимый элемент в формуле: {node}")

        restriction = get_domain_restriction(formula)

        # Создаем данные
        if restriction is not None:
            if 'log' in formula or 'log2' in formula or 'log10' in formula:
                x = np.linspace(0.01, 10, 500)  # Для логарифмов избегаем x <= 0
            elif 'sqrt' in formula:
                x = np.linspace(0, 10, 500)  # Для sqrt избегаем x < 0
            else:
                x = np.linspace(-10, 10, 500)
        else:
            x = np.linspace(-10, 10, 500)


        try:
            y = np.array([eval_expr(tree.body, x_val) for x_val in x])
        except Exception as e:
            return None, f"Ошибка при вычислении выражения: {str(e)}"


        if np.isnan(y).any() or np.isinf(y).any():
            error_msg = "Результат содержит NaN или inf значения."
            if restriction is not None:
                error_msg += " Возможно, функция определена только для определённых значений x (например, логарифм — для x > 0, квадратный корень — для x >= 0)."
            return None, error_msg

        # Строим график с помощью Matplotlib
        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.title(f"{func_type}: {formula}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        # Сохраняем график в байтовый поток
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        plt.close()  # Закрываем фигуру для освобождения памяти
        return buf.getvalue(), None  # Возвращаем байтовое представление и None

    except Exception as e:
        error_message = f"Неверная формула: {str(e)}\nПример: {BaseHandler.FUNCTION_TYPES.get(func_type, '')}"
        logger.exception("Ошибка при создании графика:")
        return None, error_message  # Возвращаем None и ошибку