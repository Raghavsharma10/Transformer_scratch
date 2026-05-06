def unmute():
	'''Unmute the volume.

	Unmutes the system volume.

	Note:
			On some systems, volume is restored to its previous level after unmute, or set to 100.
	'''

	if system.get_name() == 'windows':
		# TODO: Implement volume for Windows. Looks like WinAPI is the
		# solution...
		pass

	elif system.get_name() == 'mac':
		sp.Popen(['osascript', '-e', 'set volume output muted false']).wait()

	else:
		# Linux/Unix
		if unix_is_pulseaudio_server():
			sp.Popen(['amixer', '--quiet', '-D', 'pulse', 'sset',
					  'Master', 'unmute']).wait()  # sset is *not* a typo

		else:
			sp.Popen(['amixer', '--quiet', 'sset', 'Master', 'unmute']).wait()