def get_version():
	""" Gets the current version of the package.
	"""
	version_py = os.path.join(os.path.dirname(__file__), 'deepgram', 'version.py')
	with open(version_py, 'r') as fh:
		for line in fh:
			if line.startswith('__version__'):
				return line.split('=')[-1].strip().replace('"', '')
	raise ValueError('Failed to parse version from: {}'.format(version_py))