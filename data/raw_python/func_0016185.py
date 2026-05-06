def get_google_links(limit, params, headers):
	"""
	function to fetch links equal to limit

	every Google search result page has a start index.
	every page contains 10 search results.
	"""
	links = []
	for start_index in range(0, limit, 10):
		params['start'] = start_index
		resp = s.get("https://www.google.com/search", params = params, headers = headers)
		page_links = scrape_links(resp.content, engine = 'g')
		links.extend(page_links)
	return links[:limit]