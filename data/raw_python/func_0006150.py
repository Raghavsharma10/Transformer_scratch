def _normalize(self, string):
		''' Returns a sanitized string. '''

		string = string.replace(u'\xb7', '')
		string = string.replace(u'\xa0', ' ')
		string = string.replace('selten: ', '')
		string = string.replace('Alte Rechtschreibung', '')
		string = string.strip()
		return string