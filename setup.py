from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

name = 'train'

setup(
    name=name,
    version='0.0.4',
    description=long_description.splitlines()[0],
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ravindra Marella',
    url='https://github.com/marella/{}'.format(name),
    project_urls={
        'Documentation': 'https://marella.github.io/{}/'.format(name),
        'Source Code': 'https://github.com/marella/{}'.format(name),
    },
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': [
            'gym',
            'tensorflow',
            'matplotlib',
            'pytest',
        ],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='{} reinforcement-learning machine-learning gym tensorflow'.
    format(name),
)
