from setuptools import setup, find_packages
from os import path, listdir

from distutils.core import setup, Extension

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

try: # for pip >= 10
    from pip._internal.req import parse_requirements


except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements


def find_sources(root_dir):
    sources = []
    for file in listdir(root_dir):
        _, ext = path.splitext(file)
        
        if ext in [".cpp", ".cu"]:
            sources.append(path.join(root_dir, file))

    return sources


def make_extension(name, package):
    
    pkg_path = path.join(package, name)
    
    return Extension(
        name="{}.{}._backend".format(package, name),
        sources=find_sources(path.join("third_party", name, "src")),
        include_dirs=[path.join("third_party", name, "include")]
        )


def make_pybind_extension(name, package):
    
    pkg_path = path.join(package, name)
    
    return Pybind11Extension(
        name="{}.{}._backend".format(package, name),
        sources=find_sources(path.join(here, "third_party", name, "src")),
        include_dirs=[path.join(here, "third_party", name, "include")]
        )    
    
    
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='pyVO',
    author="Tarek BOUAMER",
    author_email="tarekbouamer199147@gmail.com",
    description="",
    version="0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/Tarekbouamer",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    # Versioning

    # Requirements
    setup_requires=["setuptools_scm"],
    python_requires=">=3, <4",

    # Package description
    packages=[
        "src",
        "src.configuration",

        "src.datasets",
        "src.features",
        
        "src.utils",
        "src.utils.orbslam2_features",
        
        "src.vo",
    ],
    ext_modules=[
        # make_pybind_extension("orbslam2_features", "src.utils"),
    ],
    #cmdclass={"build_ext": build_ext},
    #include_package_data=True,
)
