def decorate(self, view):
		""" Wraps a view function to return formatted RDF graphs
		    Uses content negotiation to serialize the graph to the client-preferred format
		    Passes other content through unmodified
		"""
		from functools import wraps

		@wraps(view)
		def decorated(*args, **kwargs):
			response = view(*args, **kwargs)
			accept = self.get_accept()
			return self.output(response, accept)
		return decorated