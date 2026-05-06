def cache_clear(self):
		"""Purge the local store of all cached function information."""
		with self.cache_lock:
			cursor = self.cache_db.cursor()
			cursor.execute('DELETE FROM cache')
			self.cache_db.commit()
		self.logger.info('the RPC cache has been purged')
		return