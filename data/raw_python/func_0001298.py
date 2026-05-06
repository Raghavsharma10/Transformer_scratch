def decide(self, accepts, context_aware=False):
		""" Returns what (mimetype,format) the client wants to receive
		    Parses the given Accept header and picks the best one that
		    we know how to output
		    Returns (mimetype, format)
		    An empty Accept will default to rdf+xml
		    An Accept with */* use rdf+xml unless a better match is found
		    An Accept that doesn't match anything will return (None,None)
		    context_aware=True will allow nquad serialization
		"""
		mimetype = self.decide_mimetype(accepts, context_aware)
		# return what format to serialize as
		if mimetype is not None:
			return (mimetype, self.get_serialize_format(mimetype))
		else:
			# couldn't find a matching mimetype for the Accepts header
			return (None, None)