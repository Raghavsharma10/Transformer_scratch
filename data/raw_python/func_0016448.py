def template_to_base_path(template, google_songs):
	"""Get base output path for a list of songs for download."""

	if template == os.getcwd() or template == '%suggested%':
		base_path = os.getcwd()
	else:
		template = os.path.abspath(template)
		song_paths = [template_to_filepath(template, song) for song in google_songs]
		base_path = os.path.dirname(os.path.commonprefix(song_paths))

	return base_path