def _normalize(self, string):
		''' Returns a sanitized string. '''

		string = string.replace(u'\xa0', '')
		string = string.strip()
		return string