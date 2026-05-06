def scrape_links(html, engine):
	"""
	function to scrape file links from html response
	"""
	soup = BeautifulSoup(html, 'lxml')
	links = []

	if engine == 'd':
		results = soup.findAll('a', {'class': 'result__a'})
		for result in results:
			link = result.get('href')[15:]
			link = link.replace('/blob/', '/raw/')
			links.append(link)

	elif engine == 'g':
		results = soup.findAll('h3', {'class': 'r'})   	
		for result in results:
			link = result.a['href'][7:].split('&')[0]
			link = link.replace('/blob/', '/raw/')
			links.append(link)

	return links