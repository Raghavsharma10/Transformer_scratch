def bencode(data, f=None):
	"""
	Writes a serializable data piece to f
	The order of tests is nonarbitrary,
	as strings and mappings are iterable.
	
	If f is None, it writes to a byte buffer
	and returns a bytestring
	"""
	if f is None:
		f = BytesIO()
		_bencode_to_file(data, f)
		return f.getvalue()
	else:
		_bencode_to_file(data, f)