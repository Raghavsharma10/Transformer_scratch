def plural(self):
		''' Tries to scrape the plural version from uitmuntend.nl. '''

		element = self._first('NN')
		if element:
			element = element.split('\r\n')[0]
			if ' | ' in element:
				# This means there is a plural
				singular, plural = element.split(' | ')
				return [plural.split(' ')[1]]
			else:
				# This means there is no plural
				return ['']
		return [None]