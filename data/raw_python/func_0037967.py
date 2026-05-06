def list_items(system_wide=False):

	'''List startup programs.

	List the programs set to run at startup.

	Args:
		system_wide (bool): Gets the programs that run at system-wide startup.

	Returns:
		list: A list of dictionaries in this format:

			.. code-block:: python

				{
				  'name': 'The name of the entry.',
				  'command': 'The command used to run it.'
				}
	'''

	desktop_env = system.get_name()

	result = []

	if desktop_env == 'windows':
		sys_startup_dir = os.path.join(winreg.ExpandEnvironmentStrings('%PROGRAMDATA%'), 'Microsoft\\Windows\\Start Menu\\Programs\\Startup')
		user_startup_dir = os.path.join(get_config_dir()[0], 'Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup')

		startup_dir = sys_startup_dir if system_wide else user_startup_dir

		for file in os.listdir(startup_dir):
			file_path = os.path.join(startup_dir, file)

			result.append({ 'name': file, 'command': os.path.join(startup_dir, file) })

	elif desktop_env == 'mac':
		items_list = system.get_cmd_out('launchtl list | awk \'{print $3}\'')
		for item in items_list.split('\n'):
			# launchd stores each job as a .plist file (pseudo-xml)
			launchd_plist_paths = ['~/Library/LaunchAgents',
									'/Library/LaunchAgents',
									'/Library/LaunchDaemons',
									'/System/Library/LaunchAgents',
									'/System/Library/LaunchDaemons']

			for path in launchd_plist_paths:
				if item + '.plist' in os.listdir(path):
					plist_file = os.path.join(path, item + '.plist')

			# Parse the plist
			if sys.version_info.major == 2:
				plist_parsed = plistlib.readPlist(plist_file)
			else:
				with open(plist_file) as f:
					plist_parsed = plistlib.load(f)

			if 'Program' in plist_parsed:
				cmd = plist_parsed['Program']

				if 'ProgramArguments' in plist_parsed:
					cmd += ' '.join(plist_parsed['ProgramArguments'])

			elif 'ProgramArguments' in plist_parsed:
				cmd = ' '.join(plist_parsed['ProgramArguments'])

			else:
				cmd = ''

			result.append({ 'name': item, 'command': cmd })

		# system-wide will be handled by running the above as root
		# which will auto-happen if current process is root.

	else:
		# Linux/Unix

		# CLI
		profile = os.path.expanduser('~/.profile')

		if os.path.isfile(profile):
			with open(profile) as f:
				for line in f:
					if system.is_in_path(line.lstrip().split(' ')[0]):
						cmd_name = line.lstrip().split(' ')[0]

						result.append({ 'name': cmd_name, 'command': line.strip() })

		# /etc/profile.d
		if system_wide:
			if os.path.isdir('/etc/profile.d'):
				for file in os.listdir('/etc/profile.d'):
					file_path = os.path.join('/etc/profile.d', file)
					result.append({ 'name': file, 'command': 'sh %s' % file_path })

		# GUI

		try:
			startup_dir = directories.get_config_dir('autostart', system_wide=system_wide)[0]

			for file in os.listdir(startup_dir):
				file_parsed = desktopfile.parse(os.path.join(startup_dir, file))

				if 'Name' in file_parsed:
					name = file_parsed['Name']

				else:
					name = file.replace('.desktop', '')

				if 'Exec' in file_parsed:
					if file_parsed['Terminal']:
						cmd = applications.terminal(exec_=file_parsed['Exec'],
														return_cmd=True)
					else:
						cmd = file_parsed['Exec']

				else:
					cmd = ''

				if not file_parsed.get('Hidden', False):
					result.append({ 'name': name, 'command': cmd })

		except IndexError:
			pass

	return result