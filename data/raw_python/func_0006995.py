def content_edge_check(self, url):
		"""Retrieve headers and MD5 hash of the content for a particular url from each Fastly edge server."""
		prefixes = ["http://", "https://"]
		for prefix in prefixes:
			if url.startswith(prefix):
				url = url[len(prefix):]
				break
		content = self._fetch("/content/edge_check/%s" % url)
		return content