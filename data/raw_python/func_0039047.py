def __decompressContent(self, coding, pgctnt):
		"""
		This is really obnoxious
		"""
		#preLen = len(pgctnt)
		if coding == 'deflate':
			compType = "deflate"
			bits_opts = [
				-zlib.MAX_WBITS,       # deflate
				 zlib.MAX_WBITS,       # zlib
				 zlib.MAX_WBITS | 16,  # gzip
				 zlib.MAX_WBITS | 32,  # "automatic header detection"

				 0,  # Try to guess from header

				 # Try all the raw window options.
				 -8, -9, -10, -11, -12, -13, -14, -15,

				 # Stream with zlib headers
				  8,  9,  10,  11,  12,  13,  14,  15,

				 # With gzip header+trailer
				  8+16,  9+16,  10+16,  11+16,  12+16,  13+16,  14+16,  15+16,
				 # Automatic detection
				  8+32,  9+32,  10+32,  11+32,  12+32,  13+32,  14+32,  15+32,

			]

			err = None

			for wbits_val in bits_opts:
				try:
					pgctnt = zlib.decompress(pgctnt, wbits_val)
					return compType, pgctnt
				except zlib.error as e:
					err = e

			# We can't get here without err having thrown.
			raise err

		elif coding == 'gzip':
			compType = "gzip"

			buf = io.BytesIO(pgctnt)
			f = gzip.GzipFile(fileobj=buf)
			pgctnt = f.read()

		elif coding == "sdch":
			raise Exceptions.ContentTypeError("Wait, someone other then google actually supports SDCH compression (%s)?" % pgreq)

		else:
			compType = "none"

		return compType, pgctnt