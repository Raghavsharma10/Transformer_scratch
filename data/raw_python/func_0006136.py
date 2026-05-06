def articles(self):
		''' Tries to scrape the correct articles for singular and plural from de.pons.eu. '''

		result = [None, None]
		element = self._first('NN')
		if element:
			result[0] = [element.split(' ')[0].replace('(die)', '').strip()]
			if 'kein Plur' in element:
				# There is no plural
				result[1] = ['']
			else:
				# If a plural form exists, there is only one possibility
				result[1] = ['die']
		return result