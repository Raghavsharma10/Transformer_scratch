def construct(name, exec_, terminal=False, additional_opts={}):
	'''Construct a .desktop file and return it as a string.
	Create a standards-compliant .desktop file, returning it as a string.
	Args:
			name			(str) : The program's name.
			exec\_		  (str) : The command.
			terminal		(bool): Determine if program should be run in a terminal emulator or not. Defaults to ``False``.
			additional_opts (dict): Any additional fields.
	Returns:
			str: The constructed .desktop file.
	'''

	desktop_file = '[Desktop Entry]\n'

	desktop_file_dict = {
		'Name': name,
		'Exec': exec_,
		'Terminal': 'true' if terminal else 'false',
		'Comment': additional_opts.get('Comment', name)
	}

	desktop_file = ('[Desktop Entry]\nName={name}\nExec={exec_}\n'
					'Terminal={terminal}\nComment={comment}\n')

	desktop_file = desktop_file.format(name=desktop_file_dict['Name'],
									   exec_=desktop_file_dict['Exec'],
									   terminal=desktop_file_dict['Terminal'],
									   comment=desktop_file_dict['Comment'])

	if additional_opts is None:
		additional_opts = {}

	for option in additional_opts:
		if not option in desktop_file_dict:
			desktop_file += '%s=%s\n' % (option, additional_opts[option])

	return desktop_file