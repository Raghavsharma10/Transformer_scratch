def output(self, response, accepts):
		""" Formats a response from a view to handle any RDF graphs
		    If a view function returns an RDF graph, serialize it based on Accept header
		    If it's not an RDF graph, return it without any special handling
		"""
		graph = self.get_graph(response)
		if graph is not None:
			# decide the format
			mimetype, format = self.format_selector.decide(accepts, graph.context_aware)

			# requested content couldn't find anything
			if mimetype is None:
				return self.make_406_response()

			# explicitly mark text mimetypes as utf-8
			if 'text' in mimetype:
				mimetype = mimetype + '; charset=utf-8'

			# format the new response
			serialized = graph.serialize(format=format)
			response = self.make_new_response(response, mimetype, serialized)
			return response
		else:
			return response