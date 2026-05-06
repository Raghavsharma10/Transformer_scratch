def decompress(self, data):
		"""Decompresses LZ77 compressed text data"""

		decompressed = ""
		pos = 0
		while pos < len(data):
			currentChar = data[pos]
			if currentChar != self.referencePrefix:
				decompressed += currentChar
				pos += 1
			else:
				nextChar = data[pos + 1]
				if nextChar != self.referencePrefix:
					distance = self.__decodeReferenceInt(data[pos + 1 : pos + 3], 2)
					length = self.__decodeReferenceLength(data[pos + 3])
					start = len(decompressed) - distance - length
					end = start + length
					decompressed += decompressed[start : end]
					pos += self.minStringLength - 1
				else:
					decompressed += self.referencePrefix
					pos += 2

		return decompressed