def discover(language):
	''' Discovers all registered scrapers to be used for the generic scraping interface. '''

	debug('Discovering scrapers for \'%s\'...' % (language,))
	global scrapers, discovered
	for language in scrapers.iterkeys():
		discovered[language] = {}
		for scraper in scrapers[language]:
			blacklist = ['download', 'isdownloaded', 'getelements']
			methods = [method for method in dir(scraper) if method not in blacklist and not method.startswith('_') and callable(getattr(scraper, method))]
			for method in methods:
				if discovered[language].has_key(method):
					discovered[language][method].append(scraper)
				else:
					discovered[language][method] = [scraper]
	debug('%d scrapers with %d methods (overall) registered for \'%s\'.' % (len(scrapers[language]), len(discovered[language].keys()), language))