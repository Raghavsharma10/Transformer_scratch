def saveDirectory(alias):
	"""save a directory to a certain alias/nickname"""
	if not settings.platformCompatible():
		return False
	dataFile = open(settings.getDataFile(), "wb")
	currentDirectory = os.path.abspath(".")
	directory = {alias : currentDirectory}
	pickle.dump(directory, dataFile)
	speech.success(alias + " will now link to " + currentDirectory + ".")
	speech.success("Tip: use 'hallie go to " + alias + "' to change to this directory.")