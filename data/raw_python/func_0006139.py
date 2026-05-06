def translate(src, dest, word):
	''' Translates a word using Google Translate. '''

	results = []

	try:
		from textblob import TextBlob
		results.append(TextBlob(word).translate(from_lang = src, to = dest).string)
	except ImportError:
		pass

	if not results:
		return [None]
	return results