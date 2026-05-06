def parse(desktop_file_or_string):
	'''Parse a .desktop file.
	Parse a .desktop file or a string with its contents into an easy-to-use dict, with standard values present even if not defined in file.
	Args:
			desktop_file_or_string (str): Either the path to a .desktop file or a string with a .desktop file as its contents.
	Returns:
			dict: A dictionary of the parsed file.'''

	if os.path.isfile(desktop_file_or_string):
		with open(desktop_file_or_string) as f:
			desktop_file = f.read()

	else:
		desktop_file = desktop_file_or_string

	result = {}

	for line in desktop_file.split('\n'):
		if '=' in line:
			result[line.split('=')[0]] = line.split('=')[1]

	for key, value in result.items():
		if value == 'false':
			result[key] = False
		elif value == 'true':
			result[key] = True

	if not 'Terminal' in result:
		result['Terminal'] = False

	if not 'Hidden' in result:
		result['Hidden'] = False

	return result