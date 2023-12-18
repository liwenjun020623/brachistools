"""
Copright © 2023 Howard Hughes Medical Institute

Adapted by: YUAN Ruihong
"""

import setuptools
from setuptools import setup

import sys

install_deps = [
    'numpy>=1.20.0',
    'scipy',
    'scikit-image>=0.19.3',
    'natsort',
    'tensorflow',
    # 'torch>=1.6',
    'tqdm',
    # 'opencv-python-headless',
    # 'fastremap',
    # 'imagecodecs',
    'keras',
    'opencv-python'
]

gui_deps = [
    'pyqt6',
    'pyqt6.sip',
    'qtpy'
]

# docs_deps = [
#     'sphinx>=3.0',
#     'sphinxcontrib-apidoc',
#     'sphinx_rtd_theme',
#     'sphinx-argparse',
# ]

if sys.platform.startswith('win'):
    install_deps.append('spams-bin')
else:
    install_deps.append('spams')

try:
    import torch
    a = torch.ones(2, 3)
    maj_ver, min_ver, _ = torch.__version__.split('.')
    if maj_ver == '2' or int(min_ver) >= 6:
        install_deps.remove('torch>=1.6')
except:
    pass

try:
    import PyQt5
    gui_deps.remove('pyqt6')
    gui_deps.remove('pyqt6.sip')
    gui_deps.append('pyqt5')
    gui_deps.append('pyqt5.sip')
except:
    pass

try:
    import PySide2
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

try:
    import PySide6
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

authors = [
    ("Ruihong Yuan", "3190110636@zju.edu.cn"),
]
def get_authors(author_list):
    return ', '.join(name for name, _ in author_list)
def get_author_emails(author_list):
    return ', '.join(email for _, email in author_list)

setup(
    name="brachistools",
    license="BSD",
    author=get_authors(authors),
    author_email=get_author_emails(authors),
    description="breast cancer histological image segmentation and classification algorithm",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/tony-yuan33/brachistools",
    setup_requires=[
        'pytest-runner',
        'setuptools_scm',
    ],
    packages=setuptools.find_packages(),
    package_data={'': ['config.ini']},
    use_scm_version=True,
    install_requires=install_deps,
    tests_require=[
        'pytest',
    ],
    extras_require = {
        # 'doc': docs_deps,
        'gui': gui_deps,
    },
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
    entry_points = {
        'console_scripts': [
            'brachistools = brachistools.__main__:main']
    }
)
