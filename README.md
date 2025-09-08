# Folder Structure Template
Reproducibility is a cornerstone of good scientific practice, and sharing clear code is an integral part of this process. It not only facilitates identifying potential errors that sneaked into your code but also allows others to validate your methods on new datasets

One should not be ashamed or afraid of sharing code as the following quote beautifully summarizes it: 
*"If you trust the code enough that you’ve written a paper about its results, it’s certainly worthy of sharing."*
[source](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002815).

Creating understandable code begins with maintaining a clean structure and stick to [fundamental principles of good coding](https://www.appacademy.io/blog/python-coding-best-practices) practices.


This repository provides one possible way to structure a GitHub repository in the first place.
Below are some insights (not complete yet), along with a `README.md` template.

## Insights 
### Repository structure 

```
    repository_main
        ├───src
        │   ├───__init__.py
        │   ├───helpers.py
        │   └───main.py
        ├───.gitignore
        ├───README.md
        ├───Example.ipynb
        └───requirements.txt
```


### Explanation
1. **`src`**: Folder containing all code files:
    1. **`__init__.py`**: (Python-specific) allows you to import functions from one file to another.
    2. **`main.py`**: (Python-specific) orchestrates your functions into a pipeline (Dont define new functions here).
    3. **`helpers.py`**: Contains additional functions. You can also create further code files and structure them case-specifically (e.g., preprocessing.py, training.py, evaluation.py).
2. **`.gitignore`**: This file specifies files and directories that will be ignored by Git and not uploaded to the repository. This is important for keeping the repository clean of unnecessary or large files. (Note: Do not upload the virtual environment to the repository. Instead, include the environment name in the .gitignore file if it resides within your repository folder.)
3. **`Example.ipynb`** (Optional but very helpful): Use this file to explain your code step by step. It can also be used to demonstrate results.
4. **`requirements.txt`**: (Python-specific) This file lists all the libraries used in your project. It allows others to install the necessary dependencies to run your code. Ideally, specify the version (e.g., numpy==2.1.0).
5. **`README.md`**: This is the first thing others will see in your repository. It can be modified directly within the `README.md` file.  
Visual Studio Code (VSC) is very useful for this task, you can do a right click on the README.md and select "open preview", then also open the file to its site so you can see all the changes you make. Additionally, ChatGPT can assist with markdown formatting, such as creating tables, lists, adding images, etc.
 It should explain your project clearly and have a structure as seen in the Project Name chapter 


### How to add images 
open this readme file as raw to see how images can be added from a website. It is also possible to add images from within your repository.
<img src="https://www.meme-arsenal.com/memes/ae75d035f63d502ca83ef1e101431446.jpg" alt="drawing" width="200"/>




## README.md  Template
Use the following template below for your own project by deleting everything above the following line and adapting everything to your project :

(Cut here) <span style="display:inline-block; transform:rotate(270deg);">✂️</span>----------------------------------------------------------


# Project Name

**Short Description**: A brief, one-line summary of what the project does or its purpose.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Contributing](#contributing)
7. [FAQ](#faq)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)
10. [Tips and Tricks](#tips-and-tricks)

---

## Overview

Provide a detailed description of the project, including:
- **Purpose**: Why this project exists and its goals.
- **Use Cases**: Potential scenarios where this project can be applied.
- **Technologies**: List the tech stack (e.g., Python, JavaScript, Docker).

**Example**:
This repository contains a [Python] library for processing raw ultrasound RF data using advanced signal processing and machine learning techniques. The project aims to facilitate research in medical imaging.

---

## Features

- Bullet-pointed list of key features:
  - Scalable architecture
  - Interactive GUI (if applicable)
  - Extensive error handling
  - Plug-and-play modularity

---

## Installation & Usage 

**Prerequisites**:
- Specify the necessary software and versions.
  - Example: Python 3.9+, pip, GPU (if applicable).

**Steps**:
```bash
# Clone the repository
git clone https://github.com/username/repository-name.git

# Navigate to the directory
cd repository-name

# Create virtual environment with venv 
pip install -m venv .env

# Activate Virtual Environment
pip source ./.env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run code 
python ./src/main.py
```
### Example Output
Provide a brief description or screenshots of expected outputs (if applicable).

## Citation

<a id="1">[1]</a> 
Tobi et al  (2023). 
A Repository is worth 16x16 words.     
arXiv preprint arXiv:123.456
https://website.com

<a id="2">[2]</a> 
Tobi et al (2023). 
Unraveling Code structures for python      
arXiv preprint arXiv:789.101112

Do Citations like this [[1]](#1) and that [[2]](#2)
## Acknowledgments
Mention contributors, libraries, and resources used in the project.

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## (not for an open repository) Responsible

list everyone responsible for this repository  
