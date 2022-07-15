from distutils.command.build_ext import build_ext
import pathlib
from setuptools import Extension, setup
import os

# https://blog.csdn.net/weixin_41964962/article/details/121088237

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
        super().run()
    

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = f'{pathlib.Path(self.build_temp)}/{ext.name}'
        os.makedirs(build_temp, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        build__args = [
            '--config', config,
            "--", "-j8"
        ]

        os.chdir(build_temp)
        self.spawn(['cmake', f'{str(cwd)}/{ext.name}'] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build__args)
        os.chdir(str(cwd))


setup(
    name='conv',
    version='0.0.1',
    ext_modules=[CMakeExtension('.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False
)
