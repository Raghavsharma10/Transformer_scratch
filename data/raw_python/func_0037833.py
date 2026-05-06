def mute():
	'''Mute the volume.

	Mutes the volume.
	'''

	# NOTE: mute != 0 volume

	if system.get_name() == 'windows':
		# TODO: Implement volume for Windows. Looks like WinAPI is the
		# solution...
		pass

	elif system.get_name() == 'mac':
		sp.Popen(['osascript', '-e', 'set volume output muted true']).wait()

	else:
		# Linux/Unix
		if unix_is_pulseaudio_server():
			sp.Popen(['amixer', '--quiet', '-D', 'pulse', 'sset',
					  'Master', 'mute']).wait()  # sset is *not* a typo

		else:
			sp.Popen(['amixer', '--quiet', 'sset', 'Master', 'mute']).wait()