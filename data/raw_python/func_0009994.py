def tab(self, netloc=None, url=None, extra_id=None, use_tid=False):
		'''
		Get a chromium tab from the pool, optionally one that has an association with a specific netloc/URL.

		If no url or netloc is specified, the per-thread identifier will be used.
		If `extra_id` is specified, it's stringified value will be mixed into the pool key
		If `use_tid` is true, the per-thread identifier will be mixed into the pool key.

		In all cases, the tab pool is a least-recently-used cache, so the tab that has been accessed the
		least recently will be automatically closed if a new tab is requested, and there are already
		`tab_pool_max_size` tabs created.

		'''
		assert self.alive, "Chrome has been shut down! Cannot continue!"
		if not netloc and url:
			netloc = urllib.parse.urlparse(url).netloc
			self.log.debug("Getting tab for netloc: %s (url: %s)", netloc, url)
		# Coerce to string type so even if it's none, it doesn't hurt anything.
		key = str(netloc)
		if extra_id:
			key += " " + str(extra_id)
		if use_tid or not key:
			key += " " + str(threading.get_ident())

		if self.__started_pid != os.getpid():
			self.log.error("TabPooledChromium instances are not safe to share across multiple processes.")
			self.log.error("Please create a new in each separate multiprocesssing process.")
			raise RuntimeError("TabPooledChromium instances are not safe to share across multiple processes.")

		with self.__counter_lock:
			self.__active_tabs.setdefault(key, 0)
			self.__active_tabs[key] += 1
			if self.__active_tabs[key] > 1:
				self.log.warning("Tab with key %s checked out more then once simultaneously")

		try:
			lock, tab = self.__tab_cache[key]
			with lock:
				yield tab
		finally:

			with self.__counter_lock:
				self.__active_tabs[key] -= 1
				if self.__active_tabs[key] == 0:
					self.__active_tabs.pop(key)