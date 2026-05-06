def iterscrapers(self, method, mode = None):
		''' Iterates over all available scrapers. '''

		global discovered
		if discovered.has_key(self.language) and discovered[self.language].has_key(method):
			for Scraper in discovered[self.language][method]:
				yield Scraper