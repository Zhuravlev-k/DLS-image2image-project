import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from inference import prepare_generator, inference
from aiogram.types import FSInputFile

bot = Bot("6743992056:AAECagM0E6bgVgiSxGXoHL_CD8t2kENOv8Y")
dp = Dispatcher()

generator = prepare_generator()

@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    await message.answer(f"Привет, {message.from_user.first_name} \nПришли фото женского лица в анфас!")

@dp.message(F.photo)
async def inf(message: types.Message):
    await message.answer("Инференс пошёл. Ожидайте, пожалуйста")
    dload_path = f"{message.from_user.id}.jpeg"
    await message.bot.download(file=message.photo[-1].file_id, destination=dload_path)
    saved_path = inference(dload_path, generator)
    answer = FSInputFile(saved_path)
    await message.reply_photo(answer)

@dp.message(~F.photo)
async def no_photo(message: types.Message):
    await message.answer("Нет, пришли фото женского лица в анфас!")

async def main():
    await bot.delete_webhook(drop_pending_updates=True) # delete requests that came while bot was offlane
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())