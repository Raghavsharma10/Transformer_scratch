def validate_links(links):
	"""
	function to validate urls based on http(s) prefix and return code
	"""
	valid_links = []
	for link in links:
		if link[:7] in "http://" or link[:8] in "https://":
			valid_links.append(link)
	
	if not valid_links:
		print("No files found.")
		sys.exit(0)

	# checking valid urls for return code
	urls = {}
	for link in valid_links:
		if 'github.com' and '/blob/' in link:
			link = link.replace('/blob/', '/raw/')
		urls[link] = {'code': get_url_nofollow(link)}
		
	
	# printing valid urls with return code 200
	available_urls = []
	for url in urls:
		print("code: %d\turl: %s" % (urls[url]['code'], url))
		if urls[url]['code'] != 0:
			available_urls.append(url)

	return available_urls