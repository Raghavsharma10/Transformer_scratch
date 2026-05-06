def render_serialization(self, context, result):
		"""Render serialized responses."""
		
		resp = context.response
		serial = context.serialize
		match = context.request.accept.best_match(serial.types, default_match=self.default)
		result = serial[match](result)
		
		if isinstance(result, str):
			result = result.decode('utf-8')
		
		resp.charset = 'utf-8'
		resp.content_type = match
		resp.text = result
		
		return True