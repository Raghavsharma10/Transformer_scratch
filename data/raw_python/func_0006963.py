def goToDirectory(alias):
	"""go to a saved directory"""
	if not settings.platformCompatible():
		return False
	data = pickle.load(open(settings.getDataFile(), "rb"))
	try:
		data[alias]
	except KeyError:
		speech.fail("Sorry, it doesn't look like you have saved " + alias + " yet.")
		speech.fail("Go to the directory you'd like to save and type 'hallie save as " + alias + "\'")
		return
	try:
		(output, error) = subprocess.Popen(["osascript", "-e", CHANGE_DIR % (data[alias])], stdout=subprocess.PIPE).communicate()
	except:
		speech.fail("Something seems to have gone wrong. Please report this error to michaelmelchione@gmail.com.")
		return
	speech.success("Successfully navigating to " + data[alias])