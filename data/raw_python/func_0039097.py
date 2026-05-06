def parse_headers(fp, _class=http.client.HTTPMessage):
	"""Parses only RFC2822 headers from a file pointer.

	email Parser wants to see strings rather than bytes.
	But a TextIOWrapper around self.rfile would buffer too many bytes
	from the stream, bytes which we later need to read as bytes.
	So we read the correct bytes here, as bytes, for email Parser
	to parse.

	Note: Monkey-patched version to try to more intelligently determine
	header encoding

	"""
	headers = []
	while True:
		line = fp.readline(http.client._MAXLINE + 1)
		if len(line) > http.client._MAXLINE:
			raise http.client.LineTooLong("header line")
		headers.append(line)
		if len(headers) > http.client._MAXHEADERS:
			raise HTTPException("got more than %d headers" % http.client._MAXHEADERS)
		if line in (b'\r\n', b'\n', b''):
			break

	decoded_headers = decode_headers(headers)

	hstring = ''.join(decoded_headers)

	return email.parser.Parser(_class=_class).parsestr(hstring)