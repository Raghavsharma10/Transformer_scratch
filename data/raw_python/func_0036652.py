def _encode_buffer(string, f):
	"""Writes the bencoded form of the input string or bytes"""
	if isinstance(string, str):
		string = string.encode()
	f.write(str(len(string)).encode())
	f.write(_TYPE_SEP)
	f.write(string)