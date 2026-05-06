def audiosamples(language, word, key = ''):
	''' Returns a list of URLs to suitable audiosamples for a given word. '''

	from lltk.audiosamples import forvo, google

	urls = []
	urls += forvo(language, word, key)
	urls += google(language, word)
	return urls