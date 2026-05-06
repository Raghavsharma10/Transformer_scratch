def add_item(name, command, system_wide=False):

	'''Adds a program to startup.

	Adds a program to user startup.

	Args:
		name        (str) : The name of the startup entry.
		command     (str) : The command to run.
		system_wide (bool): Add to system-wide startup.

	Note:
		``system_wide`` requires superuser/admin privileges.

	'''

	desktop_env = system.get_name()

	if os.path.isfile(command):
		command_is_file = True

		if not desktop_env == 'windows':
			# Will not exit program if insufficient permissions
			sp.Popen(['chmod +x %s' % command], shell=True)

	if desktop_env == 'windows':
		import winreg
		if system_wide:
			startup_dir = os.path.join(winreg.ExpandEnvironmentStrings('%PROGRAMDATA%'), 'Microsoft\\Windows\\Start Menu\\Programs\\Startup')

		else:
			startup_dir = os.path.join(get_config_dir()[0], 'Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup')

		if not command_is_file:
			with open(os.path.join(startup_dir, name + '.bat'), 'w') as f:
				f.write(command)
		else:
			shutil.copy(command, startup_dir)

	elif desktop_env == 'mac':
		sp.Popen(['launchctl submit -l %s -- %s'] % (name, command), shell=True)
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

			with open(login_file, 'a') as f:
				f.write(command)

		else:
			try:
				desktop_file_name = name + '.desktop'

				startup_file = os.path.join(get_config_dir('autostart', system_wide=system_wide)[0], desktop_file_name)

				# .desktop files' Terminal option uses an independent method to find terminal emulator
				desktop_str = desktopfile.construct(name=name, exec_=command, additional_opts={'X-GNOME-Autostart-enabled': 'true'})

				with open(startup_file, 'w') as f:
					f.write(desktop_str)
			except:
				pass