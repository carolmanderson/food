from setuptools import setup

setup(name=food_tools,
      version=0.0,
      description='Code for NLP projects',
      author="Carol Anderson",
      packages=['food_tools.data_prep', 'food_tools.training'],
      package_dir={'food_tools.data_prep': 'src/data_prep',
            'food_tools.training': 'src/training'},
      install_requires=[
          'numpy'
      ])