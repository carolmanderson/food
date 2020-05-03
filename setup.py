from setuptools import setup

setup(name='food_tools',
      version=0.4,
      description='Code for NLP projects',
      author="Carol Anderson",
      packages=['food_tools', 'food_tools.data_prep', 'food_tools.training',
                'food_tools.evaluation'],
      package_dir={'food_tools': 'src', 'food_tools.data_prep': 'src/data_prep',
            'food_tools.training': 'src/training', 'food_tools.evaluation':
                         'src/evaluation'},
      install_requires=[
          'numpy', 'sklearn'
      ])