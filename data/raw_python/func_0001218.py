def decorate(self, app):
		""" Wraps a WSGI application to return formatted RDF graphs
		    Uses content negotiation to serialize the graph to the client-preferred format
		    Passes other content through unmodified
		"""
		from functools import wraps

		@wraps(app)
		def decorated(environ, start_response):
			# capture any start_response from the app
			app_response = {}
			app_response['status'] = "200 OK"
			app_response['headers'] = []
			app_response['written'] = BytesIO()
			def custom_start_response(status, headers, *args, **kwargs):
				app_response['status'] = status
				app_response['headers'] = headers
				app_response['args'] = args
				app_response['kwargs'] = kwargs
				return app_response['written'].write
			returned = app(environ, custom_start_response)

			# callbacks from the serialization
			def set_http_code(status):
				app_response['status'] = str(status)
			def set_header(header, value):
				app_response['headers'] = [(h,v) for (h,v) in app_response['headers'] if h.lower() != header.lower()]
				app_response['headers'].append((header, value))
			def set_content_type(content_type):
				set_header('Content-Type', content_type)

			# do the serialization
			accept = environ.get('HTTP_ACCEPT', '')
			new_return = self.output(returned, accept, set_http_code, set_content_type)

			# set the Vary header
			vary_headers = (v for (h,v) in app_response['headers'] if h.lower() == 'vary')
			vary_elements = list(itertools.chain(*[v.split(',') for v in vary_headers]))
			vary_elements = list(set([v.strip() for v in vary_elements]))
			if '*' not in vary_elements and 'accept' not in (v.lower() for v in vary_elements):
				vary_elements.append('Accept')
				set_header('Vary', ', '.join(vary_elements))

			# pass on the result to the parent WSGI server
			parent_writer = start_response(app_response['status'],
			                               app_response['headers'],
			                               *app_response.get('args', []),
			                               **app_response.get('kwargs', {}))
			written = app_response['written'].getvalue()
			if len(written) > 0:
				parent_writer(written)
			return new_return
		return decorated