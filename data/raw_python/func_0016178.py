def download_content_gui(**args):
	"""
	function to fetch links and download them
	"""
	global row

	if not args ['directory']:
		args ['directory'] = args ['query'].replace(' ', '-')

	root1 = Frame(root)
	t1 = threading.Thread(target = search_function,  args = (root1,
						  args['query'], args['website'], args['file_type'], args['limit'],args['option']))
	t1.start()
	task(root1)
	t1.join()

	#new frame for progress bar 
	row = Frame(root)
	row.pack()
	if args['parallel']:
		download_parallel_gui(row, links,  args['directory'], args['min_file_size'], 
								 args['max_file_size'], args['no_redirects'])
	else:
		download_series_gui(row, links, args['directory'], args['min_file_size'],
								 args['max_file_size'], args['no_redirects'])