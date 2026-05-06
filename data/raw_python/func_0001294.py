def get_default_mimetype(self):
		""" Returns the default mimetype """
		mimetype = self.default_mimetype
		if mimetype is None:	# class inherits from module default
			mimetype = DEFAULT_MIMETYPE
		if mimetype is None:	# module is set to None?
			mimetype = 'application/rdf+xml'
		return mimetype