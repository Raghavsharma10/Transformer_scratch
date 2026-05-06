def wrapcache(timeout = -1, adapter = MemoryAdapter):
	'''
	the Decorator to cache Function.
	'''
	def _wrapcache(function):
		@wraps(function)
		def __wrapcache(*args, **kws):
			hash_key = _wrap_key(function, args, kws)
			try:
				adapter_instance = adapter()
				return pickle.loads(adapter_instance.get(hash_key))
			except CacheExpiredException:
				#timeout
				value = function(*args, **kws)
				set(hash_key, value, timeout, adapter)
				return value
		return __wrapcache
	return _wrapcache