def render_binary(self, context, result):
		"""Return binary responses unmodified."""
		context.response.app_iter = iter((result, ))  # This wraps the binary string in a WSGI body iterable.
		return True