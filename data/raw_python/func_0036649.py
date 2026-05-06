def _readuntil(f, end=_TYPE_END):
	"""Helper function to read bytes until a certain end byte is hit"""
	buf = bytearray()
	byte = f.read(1)
	while byte != end:
		if byte == b'':
			raise ValueError('File ended unexpectedly. Expected end byte {}.'.format(end))
		buf += byte
		byte = f.read(1)
	return buf