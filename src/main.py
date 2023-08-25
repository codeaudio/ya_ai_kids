"""Хендлеры и старт бота"""

import logging
import random
from asyncio import to_thread
from functools import partial

import wikipedia
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.utils import executor

from containers import Conf, StableDiffusion, Setup, Storage, LangTranslator
from states import Image, Wiki, Voice
from utils import save_to_buffer

dp = Setup.dp
bot = Setup.bot
start_keyboard = Setup.text_to_image_keyboard


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    """Старт с коммандами и приветствием"""
    await message.reply(
        ('Привет! Бот умеет: \n'
         '1. Генерировать изображение из текста \n '
         '2. Генерировать мое последнее селфи \n '
         '3. Генерировать мое фото из старшей школы \n'
         '4. Искать информацию в wiki /wiki \n'
         '5. Преобразовывать текст в аудио /text_to_voice \n'
         '6. Отдавать мое голосовое с рассказом /my_story \n'
         '7. Репо проекта /repo'),
        reply_markup=start_keyboard
    )


@dp.message_handler(commands=['help'])
async def bot_help(message: types.Message):
    """Помощь"""
    await bot.send_message(message.chat.id, f'Почта для связи {Conf.EMAIL}')


@dp.message_handler(commands=['repo'])
async def repo(message: types.Message):
    """Ссылка на репо проекта"""
    await bot.send_message(message.chat.id, Conf.REPO)


async def generate_image(message: types.Message, user_id: int):
    """Создание изображения из текста"""
    try:
        text = await to_thread(LangTranslator.translator.translate, message.text)
        image = await StableDiffusion.text_to_image(text)
        img_byte_arr = await to_thread(save_to_buffer, partial(image.save, format='PNG'))
        Storage.users_image_wait_list.remove(user_id)
        await bot.send_photo(user_id, img_byte_arr, reply_to_message_id=message.message_id)
    except Exception as e:
        logging.error(e)
        Storage.users_image_wait_list.remove(user_id)
        await message.reply('Не удалось создать изображение. Попробуйте позже')


@dp.message_handler(text=[Conf.MY_LAST_PHOTO_KB_NAME, Conf.SCHOOL_PHOTO_KB_NAME])
async def generate_my_photo(message: types.Message):
    """Создание изображения по заранне заданному тексту"""
    user_id = message.from_user.id
    if user_id in Storage.users_image_wait_list:
        return await message.reply('Вы уже отправляли запрос на генерацию изображения. Ожидайте')
    Storage.users_image_wait_list.add(user_id)
    await bot.send_message(user_id, 'Генерация изображения может занять время. Ожидайте')
    await generate_image(message, user_id)


@dp.message_handler(text=[Conf.TEXT_TO_IMAGE_KB_NAME])
async def get_text_to_image(message: types.Message):
    """Ожидание ввода текста для преобразования в изображение"""
    user_id = message.from_user.id
    if user_id in Storage.users_image_wait_list:
        return await message.reply('Вы уже отправляли запрос на генерацию изображения. Ожидайте')
    Storage.users_image_wait_list.add(user_id)
    await Image.text.set()
    await message.reply('Введите текст по которому сгенерируется изображение')


@dp.message_handler(state=Image.text)
async def generate_text_to_image(message: types.Message, state: FSMContext):
    """Создание изображения из текста пользователя"""
    await state.finish()
    user_id = message.from_user.id
    await message.reply('Генерация изображения может занять время. Ожидайте')
    await generate_image(message, user_id)


@dp.message_handler(commands=['wiki'])
async def get_wiki_text(message: types.Message):
    """Ожидание ввода текста для поиско в wiki"""
    await Wiki.text.set()
    await message.reply('Введите текст для поиска в базе знаний')


@dp.message_handler(state=Wiki.text)
async def wiki_search(message: types.Message, state: FSMContext):
    """Поиск информации в wiki"""
    await state.finish()
    try:
        await message.reply(await to_thread(wikipedia.summary, message.text))
    except wikipedia.exceptions.PageError:
        await message.reply('Информация не найдена')
    except wikipedia.exceptions.WikipediaException:
        await message.reply('Произошла ошибка. Попробуйте позже')


@dp.message_handler(commands=['text_to_voice'])
async def get_voice_text(message: types.Message):
    """Ожидание ввода текста для преобразования текста в аудио"""
    await Voice.text.set()
    await message.reply('Введите текст для преобразования в аудио')


async def generate_audio_from_text(message: types.Message, state: FSMContext, text: str):
    """Преобразование текста в аудио"""
    await state.finish()
    gtts = await to_thread(LangTranslator.text_to_audio, text)
    audio_byte_arr = await to_thread(save_to_buffer, gtts.write_to_fp)
    await bot.send_voice(
        message.from_user.id, audio_byte_arr, reply_to_message_id=message.message_id
    )


@dp.message_handler(state=Voice.text)
async def text_to_voice(message: types.Message, state: FSMContext):
    """Преобразвоание текста в аудио"""
    await generate_audio_from_text(message, state, message.text)


@dp.message_handler(commands=['my_story'])
async def my_story(message: types.Message, state: FSMContext):
    """Моя рандомная история в аудио"""
    my_story_text = Conf.MY_STORIES[random.randint(0, len(Conf.MY_STORIES) - 1)]
    await generate_audio_from_text(message, state, my_story_text)


if __name__ == '__main__':
    executor.start_polling(dp, relax=0.4)
