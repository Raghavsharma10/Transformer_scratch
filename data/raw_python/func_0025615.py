def application(self, environ, start_response):
		"""Process a single WSGI request/response cycle.
		
		This is the WSGI handler for WebCore.  Depending on the presence of extensions providing WSGI middleware,
		the `__call__` attribute of the Application instance will either become this, or become the outermost
		middleware callable.
		
		Most apps won't utilize middleware, the extension interface is preferred for most operations in WebCore.
		They allow for code injection at various intermediary steps in the processing of a request and response.
		"""
		context = environ['wc.context'] = self.RequestContext(environ=environ)
		signals = context.extension.signal
		
		# Announce the start of a request cycle. This executes `prepare` and `before` callbacks in the correct order.
		for ext in signals.pre: ext(context)
		
		# Identify the endpoint for this request.
		is_endpoint, handler = context.dispatch(context, context.root, context.environ['PATH_INFO'])
		
		if is_endpoint:
			try:
				result = self._execute_endpoint(context, handler, signals)  # Process the endpoint.
			except Exception as e:
				log.exception("Caught exception attempting to execute the endpoint.")
				result = HTTPInternalServerError(str(e) if __debug__ else "Please see the logs.")
				
				if 'debugger' in context.extension.feature:
					context.response = result
					for ext in signals.after: ext(context)  # Allow signals to clean up early.
					raise
		
		else:  # If no endpoint could be resolved, that's a 404.
			result = HTTPNotFound("Dispatch failed." if __debug__ else None)
		
		if __debug__:
			log.debug("Result prepared, identifying view handler.", extra=dict(
					request = id(context),
					result = safe_name(type(result))
				))
		
		# Identify a view capable of handling this result.
		for view in context.view(result):
			if view(context, result): break
		else:
			# We've run off the bottom of the list of possible views.
			raise TypeError("No view could be found to handle: " + repr(type(result)))
		
		if __debug__:
			log.debug("View identified, populating response.", extra=dict(
					request = id(context),
					view = repr(view),
				))
		
		for ext in signals.after: ext(context)
		
		def capture_done(response):
			for chunk in response:
				yield chunk
			
			for ext in signals.done: ext(context)
		
		# This is really long due to the fact we don't want to capture the response too early.
		# We need anything up to this point to be able to simply replace `context.response` if needed.
		return capture_done(context.response.conditional_response_app(environ, start_response))