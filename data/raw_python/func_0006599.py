def play(song, artist=None, album=None):
	"""Tells iTunes to play a given song/artist/album - MACOSX ONLY"""

	if not settings.platformCompatible():
		return False

	if song and not artist and not album:
		(output, error) = subprocess.Popen(["osascript", "-e", DEFAULT_ITUNES_PLAY % (song, song, song)], stdout=subprocess.PIPE).communicate()
		if output:
			speech.speak("Playing " + output)
		else:
			speech.speak("Unable to find " + song + " in your library.")

	elif song and artist and not album:
		(output, error) = subprocess.Popen(["osascript", "-e", ITUNES_SONG_AND_ARTIST % (song, artist, song, artist)], stdout=subprocess.PIPE).communicate()
		if output:
			speech.speak("Playing " + output)
		else:
			speech.speak("Unable to find " + song + " in your library.")

	elif album and artist and not song:
		(output, error) = subprocess.Popen(["osascript", "-e", ITUNES_ALBUM_AND_ARTIST % (artist, album)], stdout=subprocess.PIPE).communicate()
		if output:
			speech.speak("Playing " + output)
		else:
			speech.speak("Unable to find " + song + " in your library.")

	elif album and not artist and not song:
		(output, error) = subprocess.Popen(["osascript", "-e", ITUNES_ALBUM % (album)], stdout=subprocess.PIPE).communicate()
		if output:
			speech.speak("Playing " + output)
		else:
			speech.speak("Unable to find " + song + " in your library.")

	elif artist and not album and not song:
		(output, error) = subprocess.Popen(["osascript", "-e", ITUNES_ARTIST % (artist)], stdout=subprocess.PIPE).communicate()
		if output:
			speech.speak("Playing " + output)
		else:
			speech.speak("Unable to find " + song + " in your library.")