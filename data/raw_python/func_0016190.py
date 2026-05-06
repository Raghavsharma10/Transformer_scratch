def search(query, engine='g', site="", file_type = 'pdf', limit = 10):
	"""
	main function to search for links and return valid ones
	"""
	if site == "":
		search_query = "filetype:{0} {1}".format(file_type, query)
	else:
		search_query = "site:{0} filetype:{1} {2}".format(site,file_type, query)

	headers = {
		'User Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) \
		Gecko/20100101 Firefox/53.0'
	}
	if engine == "g":
		params = {
			'q': search_query,
			'start': 0,
		}
		links = get_google_links(limit, params, headers)

	elif engine == "d":
		params = {
			'q': search_query,
		}
		links = get_duckduckgo_links(limit,params,headers)
	else:
		print("Wrong search engine selected!")
		sys.exit()
	
	valid_links = validate_links(links)
	return valid_links