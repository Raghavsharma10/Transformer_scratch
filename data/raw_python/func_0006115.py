def plural(self):
		''' Tries to scrape the plural version from vandale.nl. '''

		element = self._first('NN')
		if element:
			if re.search('meervoud: ([\w|\s|\'|\-|,]+)', element, re.U):
				results = re.search('meervoud: ([\w|\s|\'|\-|,]+)', element, re.U).groups()[0].split(', ')
				results = [x.replace('ook ', '').strip() for x in results]
				return results
			else:
				# There is no plural form
				return ['']
		return [None]