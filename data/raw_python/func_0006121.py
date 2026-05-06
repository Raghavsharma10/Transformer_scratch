def gender(self):
		''' Tries to scrape the gender for a given noun from leo.org. '''

		element = self._first('NN')
		if element:
			if re.search('([m|f|n)])\.', element, re.U):
				genus = re.findall('([m|f|n)])\.', element, re.U)[0]
				return genus