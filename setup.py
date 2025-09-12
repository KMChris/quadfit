from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys

extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3", "-std=c99", "-Wall"]

ext = Extension(
    "quadfit",
    sources=["quadfit/quadfitmodule.c"],
    include_dirs=[],
    extra_compile_args=extra_compile_args,
)

class build_ext_with_numpy(build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy as np
        for e in self.extensions:
            e.include_dirs.append(np.get_include())

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext_with_numpy},
    packages=find_packages()
)
