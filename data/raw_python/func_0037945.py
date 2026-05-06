def get_cmd_out(command):
	'''Get the output of a command.

	Gets a nice Unicode no-extra-whitespace string of the ``stdout`` of a given command.

	Args:
			command (str or list): A string of the command, or a list of the arguments (as would be used in :class:`subprocess.Popen`).

	Note:
			If ``command`` is a ``str``, it will be evaluated with ``shell=True`` i.e. in the default shell (for example, bash).

	Returns:
			str: The ``stdout`` of the command.'''

	if isinstance(command, list):
		result = sp.check_output(command)
	else:
		result = sp.check_output(command, shell=True)

	return result.decode('utf-8').rstrip()