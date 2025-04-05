import asyncio
import logging
from bot import TelegramBot, BotDispatcher
from bot.handlers import functions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Запуск бота...")

    bot = TelegramBot()
    dispatcher = BotDispatcher()


    dispatcher.dp.include_router(functions.router)
    logger.info("Бот готов к работе!")
    await dispatcher.dp.start_polling(bot.instance)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Выходим из бота")