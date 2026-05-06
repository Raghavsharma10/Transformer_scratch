def removeFile(file):
	"""remove a file"""
	if "y" in speech.question("Are you sure you want to remove " + file + "? (Y/N): "):
		speech.speak("Removing " + file + " with the 'rm' command.")
		subprocess.call(["rm", "-r", file])
	else:
		speech.speak("Okay, I won't remove " + file + ".")