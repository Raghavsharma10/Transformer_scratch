def validate_args(**args):
	"""
	function to check if input query is not None 
	and set missing arguments to default value
	"""
	if not args['query']:
		print("\nMissing required query argument.")
		sys.exit()

	for key in DEFAULTS:
		if key not in args:
			args[key] = DEFAULTS[key]

	return args