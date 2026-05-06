def cache_call_refresh(self, method, *options):
		"""
		Call a remote method and update the local cache with the result
		if it already existed.

		:param str method: The name of the remote procedure to execute.
		:return: The return value from the remote function.
		"""
		options_hash = self.encode(options)
		if len(options_hash) > 20:
			options_hash = hashlib.new('sha1', options).digest()
		options_hash = sqlite3.Binary(options_hash)

		with self.cache_lock:
			cursor = self.cache_db.cursor()
			cursor.execute('DELETE FROM cache WHERE method = ? AND options_hash = ?', (method, options_hash))
		return_value = self.call(method, *options)
		store_return_value = sqlite3.Binary(self.encode(return_value))
		with self.cache_lock:
			cursor = self.cache_db.cursor()
			cursor.execute('INSERT INTO cache (method, options_hash, return_value) VALUES (?, ?, ?)', (method, options_hash, store_return_value))
			self.cache_db.commit()
		return return_value