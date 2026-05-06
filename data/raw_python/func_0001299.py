def wants_rdf(self, accepts):
		""" Returns whether this client's Accept header indicates
		    that the client wants to receive RDF
		"""
		mimetype = mimeparse.best_match(all_mimetypes + self.all_mimetypes + [WILDCARD], accepts)
		return mimetype and mimetype != WILDCARD