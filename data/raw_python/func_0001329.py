def get_graph(cls, response):
		""" Given a Flask response, find the rdflib Graph """
		if cls.is_graph(response):	# single graph object
			return response

		if hasattr(response, '__getitem__'):	# indexable tuple
			if len(response) > 0 and \
			   cls.is_graph(response[0]):	# graph object
				return response[0]