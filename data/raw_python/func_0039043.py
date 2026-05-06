def getFileAndName(self, *args, **kwargs):
		'''
		Give a requested page (note: the arguments for this call are forwarded to getpage()),
		return the content at the target URL and the filename for the target content as a
		2-tuple (pgctnt, hName) for the content at the target URL.

		The filename specified in the content-disposition header is used, if present. Otherwise,
		the last section of the url path segment is treated as the filename.
		'''

		pgctnt, hName, mime = self.getFileNameMime(*args, **kwargs)
		return pgctnt, hName