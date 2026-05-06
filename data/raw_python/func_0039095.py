def isUTF8Strict(data):     # pragma: no cover - Only used when cchardet is missing.
	'''
	Check if all characters in a bytearray are decodable
	using UTF-8.
	'''
	try:
		decoded = data.decode('UTF-8')
	except UnicodeDecodeError:
		return False
	else:
		for ch in decoded:
			if 0xD800 <= ord(ch) <= 0xDFFF:
				return False
		return True