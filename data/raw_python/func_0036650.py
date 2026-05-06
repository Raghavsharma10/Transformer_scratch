def _decode_buffer(f):
	"""
	String types are normal (byte)strings
	starting with an integer followed by ':'
	which designates the string’s length.
	
	Since there’s no way to specify the byte type
	in bencoded files, we have to guess
	"""
	strlen = int(_readuntil(f, _TYPE_SEP))
	buf = f.read(strlen)
	if not len(buf) == strlen:
		raise ValueError(
			'string expected to be {} bytes long but the file ended after {} bytes'
			.format(strlen, len(buf)))
	try:
		return buf.decode()
	except UnicodeDecodeError:
		return buf