def getFileNameMime(self, requestedUrl, *args, **kwargs):
		'''
		Give a requested page (note: the arguments for this call are forwarded to getpage()),
		return the content at the target URL, the filename for the target content, and
		the mimetype for the content at the target URL, as a 3-tuple (pgctnt, hName, mime).

		The filename specified in the content-disposition header is used, if present. Otherwise,
		the last section of the url path segment is treated as the filename.
		'''



		if 'returnMultiple' in kwargs:
			raise Exceptions.ArgumentError("getFileAndName cannot be called with 'returnMultiple'", requestedUrl)

		if 'soup' in kwargs and kwargs['soup']:
			raise Exceptions.ArgumentError("getFileAndName contradicts the 'soup' directive!", requestedUrl)

		kwargs["returnMultiple"] = True

		pgctnt, pghandle = self.getpage(requestedUrl, *args, **kwargs)

		info = pghandle.info()
		if not 'Content-Disposition' in info:
			hName = ''
		elif not 'filename=' in info['Content-Disposition']:
			hName = ''
		else:
			hName = info['Content-Disposition'].split('filename=')[1]
			# Unquote filename if it's quoted.
			if ((hName.startswith("'") and hName.endswith("'")) or hName.startswith('"') and hName.endswith('"')) and len(hName) >= 2:
				hName = hName[1:-1]

		mime = info.get_content_type()

		if not hName.strip():
			requestedUrl = pghandle.geturl()
			hName = urllib.parse.urlsplit(requestedUrl).path.split("/")[-1].strip()

		if "/" in hName:
			hName = hName.split("/")[-1]

		return pgctnt, hName, mime