def dispatch(self, context, consumed, handler, is_endpoint):
		"""Called as dispatch descends into a tier.
		
		The base extension uses this to maintain the "current url".
		"""
		
		request = context.request
		
		if __debug__:
			log.debug("Handling dispatch event.", extra=dict(
					request = id(context),
					consumed = consumed,
					handler = safe_name(handler),
					endpoint = is_endpoint
				))
		
		# The leading path element (leading slash) requires special treatment.
		if not consumed and context.request.path_info_peek() == '':
			consumed = ['']
		
		nConsumed = 0
		if consumed:
			# Migrate path elements consumed from the `PATH_INFO` to `SCRIPT_NAME` WSGI environment variables.
			if not isinstance(consumed, (list, tuple)):
				consumed = consumed.split('/')
			
			for element in consumed:
				if element == context.request.path_info_peek():
					context.request.path_info_pop()
					nConsumed += 1
				else:
					break
		
		# Update the breadcrumb list.
		context.path.append(Crumb(handler, Path(request.script_name)))
		
		if consumed:  # Lastly, update the remaining path element list.
			request.remainder = request.remainder[nConsumed:]