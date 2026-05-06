def _normalize(self, string):
		''' Returns a sanitized string. '''

		string = super(VerbixFr, self)._normalize(string)
		string = string.replace('il; elle', 'il/elle')
		string = string.replace('ils; elles', 'ils/elles')
		string = string.strip()
		return string