def get_username():
	"""
	Try to retrieve the username from a variety of sources.  First the
	environment variable LOGNAME is tried, if that is not set the
	environment variable USERNAME is tried, if that is not set the
	password database is consulted (only on Unix systems, if the import
	of the pwd module succeeds), finally if that fails KeyError is
	raised.
	"""
	try:
		return os.environ["LOGNAME"]
	except KeyError:
		pass
	try:
		return os.environ["USERNAME"]
	except KeyError:
		pass
	try:
		import pwd
		return pwd.getpwuid(os.getuid())[0]
	except (ImportError, KeyError):
		raise KeyError