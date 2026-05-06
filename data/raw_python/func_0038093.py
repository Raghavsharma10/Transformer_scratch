def open_file_with_default_program(file_path,
								   background=False, return_cmd=False):
	'''Opens a file with the default program for that type.

	Open the file with the user's preferred application.

	Args:
			file_path  (str) : Path to the file to be opened.
			background (bool): Run the program in the background, instead of waiting for completion. Defaults to ``False``.
			return_cmd (bool): Returns the command to run the program (str) instead of running it. Defaults to ``False``.

	Returns:
			str: Only if ``return_cmd``, the command to run the program is returned instead of running it. Else returns nothing.
	'''

	desktop_env = system.get_name()

	if desktop_env == 'windows':
		open_file_cmd = 'explorer.exe ' + "'%s'" % file_path

	elif desktop_env == 'mac':
		open_file_cmd = 'open ' + "'%s'" % file_path

	else:
		file_mime_type = system.get_cmd_out(
			['xdg-mime', 'query', 'filetype', file_path])
		desktop_file = system.get_cmd_out(
			['xdg-mime', 'query', 'default', file_mime_type])
		open_file_cmd = desktopfile.execute(desktopfile.locate(
			desktop_file)[0], files=[file_path], return_cmd=True)

	if return_cmd:
		return open_file_cmd

	else:
		def_program_proc = sp.Popen(open_file_cmd, shell=True)

		if not background:
			def_program_proc.wait()