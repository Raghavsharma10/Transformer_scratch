def articles(self):
		''' Tries to scrape the correct articles for singular and plural from uitmuntend.nl. '''

		result = [None, None]
		element = self._first('NN')
		if element:
			element = element.split('\r\n')[0]
			if ' | ' in element:
				# This means there is a plural
				singular, plural = element.split(' | ')
				singular, plural = singular.strip(), plural.strip()
			else:
				# This means there is no plural
				singular, plural = element.strip(), ''
				result[1] = ''
			if singular:
				result[0] = singular.split(' ')[0].split('/')
			if plural:
				result[1] = plural.split(' ')[0].split('/')
		return result