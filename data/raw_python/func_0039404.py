def chromiumContext(self, url, extra_tid=None):
		'''
		Return a active chromium context, useable for manual operations directly against
		chromium.

		The WebRequest user agent and other context is synchronized into the chromium
		instance at startup, and changes are flushed back to the webrequest instance
		from chromium at completion.
		'''
		assert url is not None, "You need to pass a URL to the contextmanager, so it can dispatch to the correct tab!"


		if extra_tid is True:
			extra_tid = threading.get_ident()

		return self._chrome_context(url, extra_tid=extra_tid)