def humanize(iso639):
	''' Converts ISO639 language identifier to the corresponding (human readable) language name. '''

	for i, element in enumerate(LANGUAGES):
		if element[1] == iso639 or element[2] == iso639:
			return element[0]
	return None