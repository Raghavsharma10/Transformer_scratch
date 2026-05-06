def render_none(self, context, result):
		"""Render empty responses."""
		context.response.body = b''
		del context.response.content_length
		return True