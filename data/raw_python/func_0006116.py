def miniaturize(self):
		''' Tries to scrape the miniaturized version from vandale.nl. '''

		element = self._first('NN')
		if element:
			if re.search('verkleinwoord: (\w+)', element, re.U):
				return re.findall('verkleinwoord: (\w+)', element, re.U)
			else:
				return ['']
		return [None]