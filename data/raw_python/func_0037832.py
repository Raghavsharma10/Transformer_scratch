def increase_volume(percentage):
	'''Increase the volume.

	Increase the volume by a given percentage.

	Args:
			percentage (int): The percentage (as an integer between 0 and 100) to increase the volume by.

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
		volume_int = percentage / 10
		old_volume = get()

		new_volume = old_volume + volume_int

		if new_volume > 10:
			new_volume = 10

		set_volume(new_volume * 10)

	else:
		# Linux/Unix
		formatted = '%d%%+' % percentage
		# + or - increases/decreases in amixer

		sp.Popen(['amixer', '--quiet', 'sset', 'Master', formatted]).wait()