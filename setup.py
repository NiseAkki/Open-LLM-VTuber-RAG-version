from setuptools import setup, find_packages

setup(
    name="open-llm-vtuber",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pytest",
        "openai",
        "torch",
        "ollama",
        "pyyaml",
        "loguru",
        "python-dotenv",
        "pypdf>=3.0.0",
        "numpy",
        "tqdm",
        "regex",
        "beautifulsoup4",
        "lxml",
    ],
    python_requires=">=3.8",
) 