def replace_graph(cls, response, serialized):
		""" Replace the rdflib Graph in a Flask response """
		if cls.is_graph(response):	# single graph object
			return serialized

		if hasattr(response, '__getitem__'):	# indexable tuple
			if len(response) > 0 and \
			   cls.is_graph(response[0]):	# graph object
				return (serialized,) + response[1:]
		return response