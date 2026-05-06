def cache_call(self, method, *options):
		"""
		Call a remote method and store the result locally. Subsequent
		calls to the same method with the same arguments will return the
		cached result without invoking the remote procedure. Cached results are
		kept indefinitely and must be manually refreshed with a call to
		:py:meth:`.cache_call_refresh`.

		:param str method: The name of the remote procedure to execute.
		:return: The return value from the remote function.
		"""
		options_hash = self.encode(options)
		if len(options_hash) > 20:
			options_hash = hashlib.new('sha1', options_hash).digest()
		options_hash = sqlite3.Binary(options_hash)

		with self.cache_lock:
			cursor = self.cache_db.cursor()
			cursor.execute('SELECT return_value FROM cache WHERE method = ? AND options_hash = ?', (method, options_hash))
			return_value = cursor.fetchone()
		if return_value:
			return_value = bytes(return_value[0])
			return self.decode(return_value)
		return_value = self.call(method, *options)
		store_return_value = sqlite3.Binary(self.encode(return_value))
		with self.cache_lock:
			cursor = self.cache_db.cursor()
			cursor.execute('INSERT INTO cache (method, options_hash, return_value) VALUES (?, ?, ?)', (method, options_hash, store_return_value))
			self.cache_db.commit()
		return return_value