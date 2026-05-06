def render_file(self, context, result):
		"""Perform appropriate metadata wrangling for returned open file handles."""
		if __debug__:
			log.debug("Processing file-like object.", extra=dict(request=id(context), result=repr(result)))
		
		response = context.response
		response.conditional_response = True
		
		modified = mktime(gmtime(getmtime(result.name)))
		
		response.last_modified = datetime.fromtimestamp(modified)
		ct, ce = guess_type(result.name)
		if not ct: ct = 'application/octet-stream'
		response.content_type, response.content_encoding = ct, ce
		response.etag = unicode(modified)
		
		result.seek(0, 2)  # Seek to the end of the file.
		response.content_length = result.tell()
		
		result.seek(0)  # Seek back to the start of the file.
		response.body_file = result
		
		return True