def remove_item(name, system_wide=False):

	'''Removes a program from startup.

	Removes a program from startup.

	Args:
		name        (str) : The name of the program (as known to the system) to remove. See :func:``list_items``.
		system_wide (bool): Remove it from system-wide startup.

	Note:
		``system_wide`` requires superuser/admin privileges.
	'''

	desktop_env = system.get_name()

	if desktop_env == 'windows':
		import winreg
		if system_wide:
			startup_dir = os.path.join(winreg.ExpandEnvironmentStrings('%PROGRAMDATA%'), 'Microsoft\\Windows\\Start Menu\\Programs\\Startup')

		else:
			startup_dir = os.path.join(directories.get_config_dir()[0], 'Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup')

		for startup_file in os.path.listdir(start_dir):
			if startup_file == name or startup_file.split('.')[0] == name:
				os.remove(os.path.join(startup_dir, startup_file))

	elif desktop_env == 'mac':
		sp.Popen(['launchctl', 'remove', name])
		# system-wide will be handled by running the above as root
		# which will auto-happen if current process is root.

	else:
		# Linux/Unix

		if desktop_env == 'unknown':
			# CLI
			if system_wide:
				login_file = '/etc/profile'
			else:
				login_file = os.path.expanduser('~/.profile')

			with open(login_file) as f:
				login_file_contents = f.read()

			final_login_file_contents = ''

			for line in login_file_contents.split('\n'):
				if line.split(' ')[0] != name:
					final_login_file_contents += line

			with open(login_file, 'w') as f:
				f.write(final_login_file_contents)

		else:
			try:
				desktop_file_name = name + '.desktop'

				startup_file = os.path.join(directories.get_config_dir('autostart', system_wide=system_wide)[0], desktop_file_name)

				if not os.path.isfile(startup_file):
					for possible_startup_file in os.listdir(directories.get_config_dir('autostart', system_wide=system_wide)[0]):
						possible_startup_file_parsed = desktopfile.parse(possible_startup_file)

						if possible_startup_file_parsed['Name'] == name:
							startup_file = possible_startup_file

				os.remove(startup_file)

			except IndexError:
				pass