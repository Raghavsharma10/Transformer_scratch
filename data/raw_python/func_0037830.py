def set_volume(percentage):
	'''Set the volume.

	Sets the volume to a given percentage (integer between 0 and 100).

	Args:
			percentage (int): The percentage (as a 0 to 100 integer) to set the volume to.

	Raises:
			ValueError: if the percentage is >100 or <0.
	'''

	if percentage > 100 or percentage < 0:
		raise ValueError('percentage must be an integer between 0 and 100')

	if system.get_name() == 'windows':
		# TODO: Implement volume for Windows. Looks like WinAPI is the
		# solution...
		pass

	elif system.get_name() == 'mac':
		# OS X uses 0-10 instead of percentage
		volume_int = percentage / 10

		sp.Popen(['osascript', '-e', 'set Volume %d' % volume_int]).wait()

	else:
		# Linux/Unix
		formatted = str(percentage) + '%'
		sp.Popen(['amixer', '--quiet', 'sset', 'Master', formatted]).wait()