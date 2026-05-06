def download_series_gui(frame, urls, directory, min_file_size, max_file_size, no_redirects):
	"""
	called when user wants serial downloading
	"""

	# create directory to save files
	if not os.path.exists(directory):
		os.makedirs(directory)
	app = progress_class(frame, urls, directory, min_file_size, max_file_size, no_redirects)