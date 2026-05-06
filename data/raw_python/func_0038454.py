def execute(desktop_file, files=None, return_cmd=False, background=False):
	'''Execute a .desktop file.
	Executes a given .desktop file path properly.
	Args:
			desktop_file (str) : The path to the .desktop file.
			files		(list): Any files to be launched by the .desktop. Defaults to empty list.
			return_cmd   (bool): Return the command (as ``str``) instead of executing. Defaults to ``False``.
			background   (bool): Run command in background. Defaults to ``False``.
	Returns:
			str: Only if ``return_cmd``. Returns command instead of running it. Else returns nothing.
	'''

	# Attempt to manually parse and execute

	desktop_file_exec = parse(desktop_file)['Exec']

	for i in desktop_file_exec.split():
		if i.startswith('%'):
			desktop_file_exec = desktop_file_exec.replace(i, '')

	desktop_file_exec = desktop_file_exec.replace(r'%F', '')
	desktop_file_exec = desktop_file_exec.replace(r'%f', '')

	if files:
		for i in files:
			desktop_file_exec += ' ' + i

	if parse(desktop_file)['Terminal']:
		# Use eval and __import__ to bypass a circular dependency
		desktop_file_exec = eval(
				('__import__("libdesktop").applications.terminal(exec_="%s",'
				 ' keep_open_after_cmd_exec=True, return_cmd=True)') %
			desktop_file_exec)

	if return_cmd:
		return desktop_file_exec

	desktop_file_proc = sp.Popen([desktop_file_exec], shell=True)

	if not background:
		desktop_file_proc.wait()