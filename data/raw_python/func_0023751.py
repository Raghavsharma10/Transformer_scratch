def expand(self, info=b"", length=32):
		'''
		Generate output key material based on an `info` value

		Arguments:
		- info - context to generate the OKM
		- length - length in bytes of the key to generate

		See the HKDF draft RFC for guidance.
		'''
		return hkdf_expand(self._prk, info, length, self._hash)