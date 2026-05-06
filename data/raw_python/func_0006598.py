def skip():
	"""Tell iTunes to skip a song"""

	if not settings.platformCompatible():
		return False

	(output, error) = subprocess.Popen(["osascript", "-e", SKIP], stdout=subprocess.PIPE).communicate()