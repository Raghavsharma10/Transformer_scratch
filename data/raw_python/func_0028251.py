def _wrap_key(function, args, kws):
	'''
	get the key from the function input.
	'''
	return hashlib.md5(pickle.dumps((_from_file(function) + function.__name__, args, kws))).hexdigest()