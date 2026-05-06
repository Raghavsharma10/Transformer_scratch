def _normalize(self, string):
		''' Returns a sanitized string. '''

		string = string.replace(u'\xb7', '')
		string = string.replace(u'\u0331', '')
		string = string.replace(u'\u0323', '')
		string = string.strip(' \n\rI.')
		return string