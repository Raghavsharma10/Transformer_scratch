def language(l):
	''' Use this as a decorator (implicitly or explicitly). '''

	# Usage: @language('en') or function = language('en')(function)

	def decorator(f):
		''' Decorator used to prepend the language as an argument. '''

		@wraps(f)
		def wrapper(*args, **kwargs):
			return f(l, *args, **kwargs)
		return wrapper

	return decorator