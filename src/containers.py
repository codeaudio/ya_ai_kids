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
        'Разница между SQL и NoSQL. SQL используется в ряляционных СУБД. Всегда есть таблица. Яркий пример - postgres.'
        ' NoSQL - это любая структура данных и иной язык запросов. Яркий пример Redis',
        'В первом классе был. Одну девочку любил! Я ей стихи писал и цветы дарил, А она меня в лоб била.'
    )
    wikipedia.set_lang("ru")


class Storage:
    users_image_wait_list = set()
    memory_storage = MemoryStorage()


class Setup:
    bot = Bot(token=Conf.TELEGRAM_BOT_TOKEN)
    dp = Dispatcher(bot, storage=Storage.memory_storage)
    text_to_image = KeyboardButton(Conf.TEXT_TO_IMAGE_KB_NAME)
    generate_my_last_photo = KeyboardButton(Conf.MY_LAST_PHOTO_KB_NAME)
    generate_school_photo = KeyboardButton(Conf.SCHOOL_PHOTO_KB_NAME)
    text_to_image_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    text_to_image_keyboard.add(text_to_image, generate_my_last_photo, generate_school_photo)


class StableDiffusion:
    if Conf.IS_MAC_MPS:
        mps = False
        mps_pipe = DiffusionPipeline.from_pretrained(**Conf.DIFFUSION_PIPELINE_CONF).to('mps')
        mps_pipe.enable_attention_slicing()

    @classmethod
    async def text_to_image(cls, text: str) -> Image:
        if Conf.IS_MAC_MPS:
            image = await cls.__text_to_image_mps(text)
        else:
            image = await cls.__text_to_image_default(text)
        return image.resize(Conf.IMAGE_SIZE)

    @classmethod
    async def __text_to_image_mps(cls, text: str) -> Image:
        is_changed = False
        if cls.mps:
            pipe = await to_thread(
                lambda: DiffusionPipeline.from_pretrained(**Conf.DIFFUSION_PIPELINE_CONF).to('cpu')
            )
            pipe.enable_attention_slicing()
        else:
            is_changed = True
            cls.mps = True
            pipe = cls.mps_pipe
        result = await to_thread(lambda prompt=text: pipe(prompt, num_inference_steps=Conf.DIFFUSION_STEP).images[0])
        if is_changed:
            cls.mps = False
        return result

    @classmethod
    async def __text_to_image_default(cls, text: str) -> Image:
        pipe = await to_thread(
            lambda: DiffusionPipeline.from_pretrained(**Conf.DIFFUSION_PIPELINE_CONF).to('cpu')
        )
        pipe.enable_attention_slicing()
        return await to_thread(
            lambda prompt=text: pipe(prompt, num_inference_steps=Conf.DIFFUSION_STEP).images[0]
        )


class LangTranslator:
    translator = MyMemoryTranslator(source='russian', target='english us')
    text_to_audio = partial(gTTS, lang='ru', slow=False)
