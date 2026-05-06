def images(language, word, n = 20, *args, **kwargs):
	''' Returns a list of URLs to suitable images for a given word.'''

	from lltk.images import google
	return google(language, word, n, *args, **kwargs)