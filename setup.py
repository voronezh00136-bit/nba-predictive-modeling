from setuptools import setup, find_packages

setup(
    name="nba-predictive-modeling",
    version="0.1.0",
    description="Machine learning pipeline for NBA game and player performance prediction",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "sqlalchemy>=2.0.0",
        "python-telegram-bot>=20.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
