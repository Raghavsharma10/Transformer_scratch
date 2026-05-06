def download_content(**args):
	"""
	main function to fetch links and download them
	"""
	args = validate_args(**args)

	if not args['directory']:
		args['directory'] = args['query'].replace(' ', '-')

	print("Downloading {0} {1} files on topic {2} from {3} and saving to directory: {4}"
		.format(args['limit'], args['file_type'], args['query'], args['website'], args['directory']))
		

	links = search(args['query'], args['engine'], args['website'], args['file_type'], args['limit'])

	if args['parallel']:
		download_parallel(links, args['directory'], args['min_file_size'], args['max_file_size'], args['no_redirects'])
	else:
		download_series(links, args['directory'], args['min_file_size'], args['max_file_size'], args['no_redirects'])