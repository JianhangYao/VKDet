from setuptools import setup, find_packages

setup(
    name="vk-det",
    version="0.1.0",
    description="VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "tests.*")),
    zip_safe=False,
)
