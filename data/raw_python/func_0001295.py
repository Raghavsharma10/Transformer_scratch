def get_wildcard_mimetype(self):
		""" Returns the mimetype if the client sends */* """
		mimetype = self.wildcard_mimetype
		if mimetype is None:	# class inherits from module default
			mimetype = WILDCARD_MIMETYPE
		if mimetype is None:	# module is set to None?
			mimetype = 'application/rdf+xml'
		return mimetype