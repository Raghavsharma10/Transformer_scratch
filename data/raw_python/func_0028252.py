def get(key, adapter = MemoryAdapter):
	'''
	get the cache value
	'''
	try:
		return pickle.loads(adapter().get(key))
	except CacheExpiredException:
		return None