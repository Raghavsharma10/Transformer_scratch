def detect_lang(path):
	"""Detect the language used in the given file."""
	blob = FileBlob(path, os.getcwd())
	if blob.is_text:
		print('Programming language of the file detected: {0}'.format(blob.language.name))
		return blob.language.name
	else:#images, binary and what-have-you won't be pasted
		print('File not a text file. Exiting...')
		sys.exit()