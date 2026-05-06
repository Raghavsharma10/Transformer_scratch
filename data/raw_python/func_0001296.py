def decide_mimetype(self, accepts, context_aware = False):
		""" Returns what mimetype the client wants to receive
		    Parses the given Accept header and returns the best one that
		    we know how to output
		    An empty Accept will default to application/rdf+xml
		    An Accept with */* use rdf+xml unless a better match is found
		    An Accept that doesn't match anything will return None
		"""
		mimetype = None
		# If the client didn't request a thing, use default
		if accepts is None or accepts.strip() == '':
			mimetype = self.get_default_mimetype()
			return mimetype

		# pick the mimetype
		if context_aware:
			mimetype = mimeparse.best_match(all_mimetypes + self.all_mimetypes + [WILDCARD], accepts)
		else:
			mimetype = mimeparse.best_match(ctxless_mimetypes + self.ctxless_mimetypes + [WILDCARD], accepts)
		if mimetype == '':
			mimetype = None

		# if browser sent */*
		if mimetype == WILDCARD:
			mimetype = self.get_wildcard_mimetype()

		return mimetype