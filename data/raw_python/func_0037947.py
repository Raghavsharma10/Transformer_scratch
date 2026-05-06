def is_in_path(program):
	'''
	Check if a program is in the system ``PATH``.

	Checks if a given program is in the user's ``PATH`` or not.

	Args:
			program (str): The program to try to find in ``PATH``.

	Returns:
			bool: Is the program in ``PATH``?
	'''

	if sys.version_info.major == 2:
		path = os.getenv('PATH')
		if os.name == 'nt':
			path = path.split(';')
		else:
			path = path.split(':')
	else:
		path = os.get_exec_path()

	for i in path:
		if os.path.isdir(i):
			if program in os.listdir(i):
				return True