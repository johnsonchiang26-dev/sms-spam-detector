"""Package definition for the SMS spam & phishing detector.

Installing in editable mode (``pip install -e .``) makes the ``src`` package
importable from anywhere (notebooks, tests, CLI scripts) without juggling
``sys.path`` hacks.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
REQUIREMENTS = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="sms-spam-detector",
    version="0.1.0",
    description="BERT fine-tuning for SMS spam & financial phishing detection.",
    author="Johnson Chiang",
    url="https://github.com/johnsonchiang26-dev/sms-spam-detector",
    packages=find_packages(exclude=("tests", "notebooks")),
    python_requires=">=3.9",
    install_requires=REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "sms-train=src.train:main",
            "sms-predict=src.predict:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
