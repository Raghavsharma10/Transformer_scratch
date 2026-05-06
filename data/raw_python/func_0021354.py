def guess_mime_type(self, path):
		"""
		Guess an appropriate MIME type based on the extension of the
		provided path.

		:param str path: The of the file to analyze.
		:return: The guessed MIME type of the default if non are found.
		:rtype: str
		"""
		_, ext = posixpath.splitext(path)
		if ext in self.extensions_map:
			return self.extensions_map[ext]
		ext = ext.lower()
		return self.extensions_map[ext if ext in self.extensions_map else '']