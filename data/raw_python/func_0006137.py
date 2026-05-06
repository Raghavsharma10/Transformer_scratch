def plural(self):
		''' Tries to scrape the plural version from pons.eu. '''

		element = self._first('NN')
		if element:
			if 'kein Plur' in element:
				# There is no plural
				return ['']
			if re.search(', ([\w|\s|/]+)>', element, re.U):
				# Plural form is provided
				return re.findall(', ([\w|\s|/]+)>', element, re.U)[0].split('/')
			if re.search(', -(\w+)>', element, re.U):
				# Suffix is provided
				suffix = re.findall(', -(\w+)>', element, re.U)[0]
				return [self.word + suffix]
			if element.endswith('->'):
				# Plural is the same as singular
				return [self.word]
		return [None]