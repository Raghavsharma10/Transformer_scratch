def copy(location):
	"""copy file or directory at a given location; can be pasted later"""
	copyData = settings.getDataFile()
	copyFileLocation = os.path.abspath(location)
	copy = {"copyLocation": copyFileLocation}
	dataFile = open(copyData, "wb")
	pickle.dump(copy, dataFile)
	speech.speak(location + " copied successfully!")
	speech.speak("Tip: use 'hallie paste' to paste this file.")