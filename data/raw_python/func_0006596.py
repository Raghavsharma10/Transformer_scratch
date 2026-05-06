def pause():
	"""Tell iTunes to pause"""

	if not settings.platformCompatible():
		return False

	(output, error) = subprocess.Popen(["osascript", "-e", PAUSE], stdout=subprocess.PIPE).communicate()