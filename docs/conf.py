from datetime import datetime

project = 'Train'
author = 'Ravindra Marella'
copyright = f'{datetime.now().year}, {author}'

html_theme = 'sphinx_rtd_theme'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

autodoc_default_options = {
    'members': True,
}

napoleon_use_ivar = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}
