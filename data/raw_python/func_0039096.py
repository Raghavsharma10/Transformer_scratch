def decode_headers(header_list):
	'''
	Decode a list of headers.

	Takes a list of bytestrings, returns a list of unicode strings.
	The character set for each bytestring is individually decoded.
	'''

	decoded_headers = []
	for header in header_list:
		if cchardet:
			inferred = cchardet.detect(header)
			if inferred and inferred['confidence'] > 0.8:
				# print("Parsing headers!", header)
				decoded_headers.append(header.decode(inferred['encoding']))
			else:
				decoded_headers.append(header.decode('iso-8859-1'))
		else:    # pragma: no cover
			# All bytes are < 127 (e.g. ASCII)
			if all([char & 0x80 == 0 for char in header]):
				decoded_headers.append(header.decode("us-ascii"))
			elif isUTF8Strict(header):
				decoded_headers.append(header.decode("utf-8"))
			else:
				decoded_headers.append(header.decode('iso-8859-1'))

	return decoded_headers