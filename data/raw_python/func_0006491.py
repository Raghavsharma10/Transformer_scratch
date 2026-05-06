def paste(location):
	"""paste a file or directory that has been previously copied"""
	copyData = settings.getDataFile()
	if not location:
		location = "."
	try:
		data = pickle.load(open(copyData, "rb"))
		speech.speak("Pasting " + data["copyLocation"] + " to current directory.")
	except:
		speech.fail("It doesn't look like you've copied anything yet.")
		speech.fail("Type 'hallie copy <file>' to copy a file or folder.")
		return
	process, error = subprocess.Popen(["cp", "-r", data["copyLocation"], location], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()
	if "denied" in process:
		speech.fail("Unable to paste your file successfully. This is most likely due to a permission issue. You can try to run me as sudo!")