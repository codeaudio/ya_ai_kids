"""Состояния"""

from aiogram.dispatcher.filters.state import StatesGroup, State


class Image(StatesGroup):
    """Состояние для создания изображения из текста пользователя"""
    text = State()


class Wiki(StatesGroup):
    """Состояние поиска в wiki"""
    text = State()


class Voice(StatesGroup):
    """Состояние Преобразвоания текста в аудио"""
    text = State()
