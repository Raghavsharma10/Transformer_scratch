def render_generator(self, context, result):
		"""Attempt to serve generator responses through stream encoding.
		
		This allows for direct use of cinje template functions, which are generators, as returned views.
		"""
		context.response.encoding = 'utf8'
		context.response.app_iter = (
				(i.encode('utf8') if isinstance(i, unicode) else i)  # Stream encode unicode chunks.
				for i in result if i is not None  # Skip None values.
			)
		return True