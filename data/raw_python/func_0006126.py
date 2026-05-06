def scrape(language, method, word, *args, **kwargs):
	''' Uses custom scrapers and calls provided method. '''

	scraper = Scrape(language, word)
	if hasattr(scraper, method):
		function = getattr(scraper, method)
		if callable(function):
			return function(*args, **kwargs)
	else:
		raise NotImplementedError('The method ' + method + '() is not implemented so far.')