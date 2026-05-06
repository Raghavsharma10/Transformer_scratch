def cached(key = None, extradata = {}):
	''' Decorator used for caching. '''

	def decorator(f):

		@wraps(f)
		def wrapper(*args, **kwargs):

			uid = key
			if not uid:
				from hashlib import md5
				arguments = list(args) + [(a, kwargs[a]) for a in sorted(kwargs.keys())]
				uid = md5(str(arguments)).hexdigest()
			if exists(uid):
				debug('Item \'%s\' is cached (%s).' % (uid, cache))
				return get(uid)
			else:
				debug('Item \'%s\' is not cached (%s).' % (uid, cache))
				result = f(*args, **kwargs)
				debug('Caching result \'%s\' as \'%s\' (%s)...' % (result, uid, cache))
				debug('Extra data: ' + (str(extradata) or 'None'))
				put(uid, result, extradata)
				return result
		return wrapper
	return decorator