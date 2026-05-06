def from_urls(cls, urllist, coltype=LIGOTimeGPS):
		"""
		Return a Cache whose entries are inferred from the URLs
		in urllist, if possible.  PFN lists will also work; for PFNs, the path
		will be absolutized and "file://" and "localhost" will be assumed
		for the schemes and hosts.
		
		The filenames must be in the format set forth by DASWG in T050017-00.
		"""
		def pfn_to_url(url):
			scheme, host, path, dummy, dummy = urlparse.urlsplit(url)
			if scheme == "": path = os.path.abspath(path)
			return urlparse.urlunsplit((scheme or "file", host or "localhost",
			                            path, "", ""))
		return cls([cls.entry_class.from_T050017(pfn_to_url(f), coltype=coltype) \
		            for f in urllist])