from distutils.command.build_ext import build_ext
from setuptools import Extension, setup


class CMakeExtension(Extension):
    def __init__(self, name, sources, *args, **kw) -> None:
        super().__init__(name, sources, *args, **kw)
