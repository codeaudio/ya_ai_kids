"""Контейнеры с данными"""

import os
from asyncio import to_thread
from functools import partial

import wikipedia
from PIL.Image import Image
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from deep_translator import MyMemoryTranslator
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
from gtts import gTTS


class Conf:
    """Конфиги"""
    load_dotenv()

    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    DIFFUSION_STEP = int(os.getenv('DIFFUSION_STEP'))
    EMAIL = os.getenv('EMAIL')
    REPO = os.getenv('REPO')
    IS_MAC_MPS = bool(int(os.getenv('IS_MAC_MPS')))
    IMAGE_SIZE = (int(os.getenv('IMAGE_SIZE')), int(os.getenv('IMAGE_SIZE')))
    DIFFUSION_PIPELINE_CONF = {
        'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        'safety_checker': None,
        'requires_safety_checker': False
    }
    TEXT_TO_IMAGE_KB_NAME = 'Изображение из текста 🌇'
    MY_LAST_PHOTO_KB_NAME = 'Мое последнее селфи'
    SCHOOL_PHOTO_KB_NAME = 'Мое фото из старшей школы'
    MY_STORIES = (
        'GPT - это прорыв последнего времени. Например, с помощью ChatGPT'
        ' можно ускорять свою работу и использовать как "умный поисковик"',
        'Разница между SQL и NoSQL. SQL используется в ряляционных СУБД.'
        ' Всегда есть таблица. Яркий пример - postgres.'
        ' NoSQL - это любая структура данных и иной язык запросов. Яркий пример Redis',
        'В первом классе был. Одну девочку любил!'
        ' Я ей стихи писал и цветы дарил, А она меня в лоб била.'
    )
    wikipedia.set_lang("ru")


class Storage:
    """Хранилище в рантайме"""
    users_image_wait_list = set()
    memory_storage = MemoryStorage()


class Setup:
    """Сетап бота"""
    bot = Bot(token=Conf.TELEGRAM_BOT_TOKEN)
    dp = Dispatcher(bot, storage=Storage.memory_storage)
    text_to_image = KeyboardButton(Conf.TEXT_TO_IMAGE_KB_NAME)
    generate_my_last_photo = KeyboardButton(Conf.MY_LAST_PHOTO_KB_NAME)
    generate_school_photo = KeyboardButton(Conf.SCHOOL_PHOTO_KB_NAME)
    text_to_image_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    text_to_image_keyboard.add(text_to_image, generate_my_last_photo, generate_school_photo)


class StableDiffusion:
    """StableDiffusion для создания изображения из текста"""
    __text_to_image = None
    __mps = False

    @classmethod
    async def text_to_image(cls, text: str) -> Image:
        """Создание изображения из текста"""
        image = await cls.__text_to_image(text)
        return image.resize(Conf.IMAGE_SIZE)

    @classmethod
    async def __text_to_image_mps(cls, text: str) -> Image:
        """Создание изображения используя gpu или cpu"""
        if cls.__mps:
            return await cls.__text_to_image_cpu(text)
        cls.__mps = True
        pipe = cls.mps_pipe
        try:
            result = await to_thread(
                lambda prompt=text: pipe(prompt, num_inference_steps=Conf.DIFFUSION_STEP).images[0]
            )
        finally:
            cls.__mps = False
        return result

    @classmethod
    async def __text_to_image_cpu(cls, text: str) -> Image:
        """Создание изображения используя cpu"""
        pipe = await to_thread(
            lambda: DiffusionPipeline.from_pretrained(**Conf.DIFFUSION_PIPELINE_CONF).to('cpu')
        )
        pipe.enable_attention_slicing()
        return await to_thread(
            lambda prompt=text: pipe(prompt, num_inference_steps=Conf.DIFFUSION_STEP).images[0]
        )

    if Conf.IS_MAC_MPS:
        mps_pipe = DiffusionPipeline.from_pretrained(**Conf.DIFFUSION_PIPELINE_CONF).to('mps')
        mps_pipe.enable_attention_slicing()
        __text_to_image = __text_to_image_mps
    else:
        __text_to_image = __text_to_image_cpu


class LangTranslator:
    """Работа с текстом"""
    translator = MyMemoryTranslator(source='russian', target='english us')
    text_to_audio = partial(gTTS, lang='ru', slow=False)
