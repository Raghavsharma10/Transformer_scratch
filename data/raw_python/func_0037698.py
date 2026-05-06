def get_config_file(program, system_wide=False):
	'''Get the configuration file for a program.

	Gets the configuration file for a given program, assuming it stores it in
	a standard location. See also :func:`get_config_dir()`.

	Args:
			program	 (str): The program for which to get the configuration file.
			system_wide (bool):Whether to get the system-wide file for the program.

	Returns:
			list: A list of all matching configuration files found.
	'''

	program_config_homes = get_config_dir(program, system_wide)
	config_homes = get_config_dir(system_wide=system_wide)
	config_files = []

	for home in config_homes:
		for sub in os.listdir(home):
			if os.path.isfile(os.path.join(home, sub)):
				if sub.startswith(program):
					config_files.append(os.path.join(home, sub))

	if not program.startswith('.'):
		config_files.extend(get_config_file('.' + program, system_wide))

	for home in program_config_homes:
		for sub in os.listdir(home):
			if os.path.isfile(os.path.join(home, sub)
							  ) and sub.startswith(program):
				config_files.append(os.path.join(home, sub))

	return config_files