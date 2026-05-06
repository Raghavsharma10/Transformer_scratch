def register(scraper):
	''' Registers a scraper to make it available for the generic scraping interface. '''

	global scrapers
	language = scraper('').language
	if not language:
		raise Exception('No language specified for your scraper.')
	if scrapers.has_key(language):
		scrapers[language].append(scraper)
	else:
		scrapers[language] = [scraper]