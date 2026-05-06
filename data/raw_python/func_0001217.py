def output(self, output, accepts, set_http_code, set_content_type):
		""" Formats a response from a WSGI app to handle any RDF graphs
		    If a view function returns a single RDF graph, serialize it based on Accept header
		    If it's not an RDF graph, return it without any special handling
		"""

		graph = Decorator._get_graph(output)
		if graph is not None:
			# decide the format
			output_mimetype, output_format = self.format_selector.decide(accepts, graph.context_aware)
			# requested content couldn't find anything
			if output_mimetype is None:
				set_http_code("406 Not Acceptable")
				return ['406 Not Acceptable'.encode('utf-8')]
			# explicitly mark text mimetypes as utf-8
			if 'text' in output_mimetype:
				output_mimetype = output_mimetype + '; charset=utf-8'

			# format the new response
			serialized = graph.serialize(format=output_format)
			set_content_type(output_mimetype)
			return [serialized]
		else:
			return output