def download_parallel_gui(root, urls, directory, min_file_size, max_file_size, no_redirects):
	"""
	called when paralled downloading is true
	"""
	global parallel

	# create directory to save files
	if not os.path.exists(directory):
		os.makedirs(directory)
	parallel = True
	app = progress_class(root, urls, directory, min_file_size, max_file_size, no_redirects)