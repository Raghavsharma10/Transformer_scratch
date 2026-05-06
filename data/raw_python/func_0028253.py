def set(key, value, timeout = -1, adapter = MemoryAdapter):
	'''
	set cache by code, must set timeout length
	'''
	if adapter(timeout = timeout).set(key, pickle.dumps(value)):
		return value
	else:
		return None