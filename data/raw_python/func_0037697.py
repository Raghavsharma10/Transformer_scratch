def get_config_dir(program='', system_wide=False):
	'''Get the configuration directory.

	Get the configuration directories, optionally for a specific program.

	Args:
			program	(str) : The name of the program whose configuration directories have to be found.
			system_wide (bool): Gets the system-wide configuration directories.

	Returns:
			list: A list of all matching configuration directories found.
	'''

	config_homes = []

	if system_wide:
		if os.name == 'nt':
			config_homes.append(
				winreg.ExpandEnvironmentStrings('%PROGRAMDATA%'))

		else:
			config_homes.append('/etc')
			config_homes.append('/etc/xdg')

			if os.name == 'darwin':
				config_homes.append('/Library')

	else:
		if os.name == 'nt':
			import winreg
			config_homes.append(
				winreg.ExpandEnvironmentStrings('%LOCALAPPDATA%'))
			config_homes.append(
				os.path.join(
					winreg.ExpandEnvironmentStrings('%APPDATA%'),
					'Roaming'))
		else:
			if os.getenv('XDG_CONFIG_HOME'):
				config_homes.append(os.getenv('XDG_CONFIG_HOME'))
			else:
				try:
					from xdg import BaseDirectory
					config_homes.append(BaseDirectory.xdg_config_home)
				except ImportError:
					config_homes.append(os.path.expanduser('~/.config'))

				config_homes.append(os.path.expanduser('~'))

				if os.name == 'darwin':
					config_homes.append(os.path.expanduser('~/Library'))

	if program:
		def __find_homes(app, dirs):

			homes = []

			for home in dirs:
				if os.path.isdir(os.path.join(home, app)):
					homes.append(os.path.join(home, app))

				if os.path.isdir(os.path.join(home, '.' + app)):
					homes.append(os.path.join(home, '.' + app))

				if os.path.isdir(os.path.join(home, app + '.d')):
					homes.append(os.path.join(home, app + '.d'))

			return homes

		app_homes = __find_homes(program, config_homes)

		# Special Cases

		if program == 'vim':
			app_homes.extend(__find_homes('vimfiles', config_homes))

		elif program == 'chrome':
			app_homes.extend(__find_homes('google-chrome', config_homes))

		elif program in ['firefox', 'thunderbird']:
			app_homes.extend(
				__find_homes(
					program, [
						os.path.expanduser('~/.mozilla')]))

		return app_homes

	return config_homes