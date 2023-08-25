from aiogram.dispatcher.filters.state import StatesGroup, State


class Image(StatesGroup):
    text = State()


class Wiki(StatesGroup):
    text = State()


class Voice(StatesGroup):
    text = State()
