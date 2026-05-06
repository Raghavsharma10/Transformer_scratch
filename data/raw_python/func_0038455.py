def locate(desktop_filename_or_name):
	'''Locate a .desktop from the standard locations.
	Find the path to the .desktop file of a given .desktop filename or application name.
	Standard locations:
			- ``~/.local/share/applications/``
			- ``/usr/share/applications``
	Args:
			desktop_filename_or_name (str): Either the filename of a .desktop file or the name of an application.
	Returns:
			list: A list of all matching .desktop files found.
	'''

	paths = [
		os.path.expanduser('~/.local/share/applications'),
		'/usr/share/applications']

	result = []

	for path in paths:
		for file in os.listdir(path):
			if desktop_filename_or_name in file.split(
					'.') or desktop_filename_or_name == file:
				# Example: org.gnome.gedit
				result.append(os.path.join(path, file))

			else:
				file_parsed = parse(os.path.join(path, file))

				try:
					if desktop_filename_or_name.lower() == file_parsed[
							'Name'].lower():
						result.append(file)
					elif desktop_filename_or_name.lower() == file_parsed[
							'Exec'].split(' ')[0]:
						result.append(file)
				except KeyError:
					pass

	for res in result:
		if not res.endswith('.desktop'):
			result.remove(res)

	if not result and not result.endswith('.desktop'):
		result.extend(locate(desktop_filename_or_name + '.desktop'))

	return result