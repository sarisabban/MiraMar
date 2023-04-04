import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
	name='MolecularTetris',
	version='1.0',
	author='Sari Sabban',
	author_email='',
	description='De novo cyclic protein polypeptide design using reinforcement learning.',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/sarisabban/MolecularTetris',
	project_urls={'Bug Tracker':'https://github.com/sarisabban/MolecularTetris/issues'},
	license='GPL-2.0',
	packages=['MolecularTetris'],
	install_requires=['numpy', 'scipy', 'gym', 'torch', 'tianshou'])
