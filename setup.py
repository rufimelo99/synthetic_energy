from setuptools import find_packages, setup


def read_requirements():
    """Read the requirements.txt file and return its contents as a list."""
    requirements_path = "requirements.txt"
    with open(requirements_path, "r") as f:
        return [line.strip() for line in f.readlines()]


setup(
    name="synthetic_energy",
    version="0.0.12",
    packages=find_packages(),
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        'synthetic_energy': ['anonymization/classifiers/*.pkl'],  # Include all .pkl files in the data directory
    },
    entry_points={
        "console_scripts": [
            # 'your_command=your_module.your_module:main',
        ],
    },
    author="Rui Melo & André Catarino",
    author_email="rufimelo99@gmail.com; andre.almeida.catarino@gmail.com",
    description="PhD work for ICSP @ FEUP, PORTO",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
