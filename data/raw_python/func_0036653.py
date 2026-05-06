def _encode_mapping(mapping, f):
	"""Encodes the mapping items in lexical order (spec)"""
	f.write(_TYPE_DICT)
	for key, value in sorted(mapping.items()):
		_encode_buffer(key, f)
		bencode(value, f)
	f.write(_TYPE_END)