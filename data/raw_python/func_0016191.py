def check_threats(**args):
	"""
	function to check input filetype against threat extensions list 
	"""
	is_high_threat = False
	for val in THREAT_EXTENSIONS.values():
		if type(val) == list:
			for el in val:
				if args['file_type'] == el:
					is_high_threat = True
					break
		else:
			if args['file_type'] == val:
				is_high_threat = True
				break
	return is_high_threat