def text_editor(file='', background=False, return_cmd=False):
	'''Starts the default graphical text editor.

	Start the user's preferred graphical text editor, optionally with a file.

	Args:
			file	   (str) : The file to be opened with the editor. Defaults to an empty string (i.e. no file).
			background (bool): Runs the editor in the background, instead of waiting for completion. Defaults to ``False``.
			return_cmd (bool): Returns the command (str) to run the editor instead of running it. Defaults to ``False``.

	Returns:
			str: Only if ``return_cmd``, the command to run the editor is returned. Else returns nothing.
	'''

	desktop_env = system.get_name()

	if desktop_env == 'windows':
		editor_cmd_str = system.get_cmd_out(
			['ftype', 'textfile']).split('=', 1)[1]

	elif desktop_env == 'mac':
		editor_cmd_str = 'open -a' + system.get_cmd_out(
				['def',
				 'read',
				 'com.apple.LaunchServices',
				 'LSHandlers'
				 '-array'
				 '{LSHandlerContentType=public.plain-text;}']
				)

	else:
		# Use def handler for MIME-type text/plain
		editor_cmd_str = system.get_cmd_out(
			['xdg-mime', 'query', 'default', 'text/plain'])

		if '\n' in editor_cmd_str:
			# Sometimes locate returns multiple results
			# use first one

			editor_cmd_str = editor_cmd_str.split('\n')[0]

	if editor_cmd_str.endswith('.desktop'):
		# We don't use desktopfile.execute() in order to have working
		# return_cmd and background

		editor_cmd_str = desktopfile.parse(
			desktopfile.locate(editor_cmd_str)[0])['Exec']

		for i in editor_cmd_str.split():
			if i.startswith('%'):
				# %-style formatters
				editor_cmd_str = editor_cmd_str.replace(i, '')

			if i == '--new-document':
				# Gedit
				editor_cmd_str = editor_cmd_str.replace(i, '')

	if file:
		editor_cmd_str += ' {}'.format(shlex.quote(file))

	if return_cmd:
		return editor_cmd_str

	text_editor_proc = sp.Popen([editor_cmd_str], shell=True)

	if not background:
		text_editor_proc.wait()