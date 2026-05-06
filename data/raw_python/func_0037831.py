def get_volume():
	'''Get the volume.

	Get the current volume.

	Returns:
			int: The current volume (percentage, between 0 and 100).
	'''

	if system.get_name() == 'windows':
		# TODO: Implement volume for Windows. Looks like WinAPI is the
		# solution...
		pass

	elif system.get_name() == 'mac':
		volume = system.get_cmd_out(
			['osascript', '-e', 'set ovol to output volume of (get volume settings); return the quoted form of ovol'])
		return int(volume) * 10

	else:
		# Linux/Unix
		volume = system.get_cmd_out(
				('amixer get Master |grep % |awk \'{print $5}\'|'
				 'sed -e \'s/\[//\' -e \'s/\]//\' | head -n1'))
		return int(volume.replace('%', ''))