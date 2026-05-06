def get_duckduckgo_links(limit, params, headers):
	"""
	function to fetch links equal to limit

	duckduckgo pagination is not static, so there is a limit on
	maximum number of links that can be scraped
	"""
	resp = s.get('https://duckduckgo.com/html', params = params, headers = headers)
	links = scrape_links(resp.content, engine = 'd')
	return links[:limit]