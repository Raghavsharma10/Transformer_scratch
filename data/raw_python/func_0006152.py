def register(cache):
	''' Registers a cache. '''

	global caches
	name = cache().name
	if not caches.has_key(name):
		caches[name] = cache