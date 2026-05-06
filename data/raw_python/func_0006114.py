def articles(self):
		''' Tries to scrape the correct articles for singular and plural from vandale.nl. '''

		result = [None, None]
		element = self._first('NN')
		if element:
			if re.search('(de|het/?de|het);', element, re.U):
				result[0] = re.findall('(de|het/?de|het);', element, re.U)[0].split('/')
			if re.search('meervoud: (\w+)', element, re.U):
				# It's a noun with a plural form
				result[1] = ['de']
			else:
				# It's a noun without a plural form
				result[1] = ['']
		return result