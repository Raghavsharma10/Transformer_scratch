def hkdf_extract(salt, input_key_material, hash=hashlib.sha512):
	'''
	Extract a pseudorandom key suitable for use with hkdf_expand
	from the input_key_material and a salt using HMAC with the
	provided hash (default SHA-512).

	salt should be a random, application-specific byte string. If
	salt is None or the empty string, an all-zeros string of the same
	length as the hash's block size will be used instead per the RFC.
	
	See the HKDF draft RFC and paper for usage notes.
	'''
	hash_len = hash().digest_size
	if salt == None or len(salt) == 0:
		salt = bytearray((0,) * hash_len)
	return hmac.new(bytes(salt), buffer(input_key_material), hash).digest()