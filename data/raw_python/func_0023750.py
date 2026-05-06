def hkdf_expand(pseudo_random_key, info=b"", length=32, hash=hashlib.sha512):
	'''
	Expand `pseudo_random_key` and `info` into a key of length `bytes` using
	HKDF's expand function based on HMAC with the provided hash (default
	SHA-512). See the HKDF draft RFC and paper for usage notes.
	'''
	hash_len = hash().digest_size
	length = int(length)
	if length > 255 * hash_len:
		raise Exception("Cannot expand to more than 255 * %d = %d bytes using the specified hash function" %\
			(hash_len, 255 * hash_len))
	blocks_needed = length // hash_len + (0 if length % hash_len == 0 else 1) # ceil
	okm = b""
	output_block = b""
	for counter in range(blocks_needed):
		output_block = hmac.new(pseudo_random_key, buffer(output_block + info + bytearray((counter + 1,))),\
			hash).digest()
		okm += output_block
	return okm[:length]