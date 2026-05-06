def _normalize(self, string):
		''' Returns a sanitized string. '''

		string = super(VerbixDe, self)._normalize(string)
		string = string.replace('sie; Sie', 'sie')
		string = string.strip()
		return string