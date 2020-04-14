from distutils.core import setup

setup(
  name = 'zsom',
  packages = ['zsom'],
  version = '1.0.1', 
  license='MIT',
  description = 'Implementation of the Algorithm Self-Organizing Maps.',
  author = 'Zaú Júlio A. Galvão',
  author_email = 'zauhdf@gmail.com',
  url = 'https://github.com/ZauJulio/ZSOM',
  download_url = 'https://github.com/ZauJulio/ZSOM/archive/v0.0.tar.gz',
  keywords = ['SOM', 'Self-Organizing Maps', 'Robust implementation'],
  install_requires=[
          'numpy'
      ],

  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)