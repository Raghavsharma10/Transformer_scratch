def compress(self, data, windowLength = None):
		"""Compresses text data using the LZ77 algorithm."""

		if windowLength == None:
			windowLength = self.defaultWindowLength

		compressed = ""
		pos = 0
		lastPos = len(data) - self.minStringLength

		while pos < lastPos:

			searchStart = max(pos - windowLength, 0);
			matchLength = self.minStringLength
			foundMatch = False
			bestMatchDistance = self.maxStringDistance
			bestMatchLength = 0
			newCompressed = None

			while (searchStart + matchLength) < pos:

				m1 = data[searchStart : searchStart + matchLength]
				m2 = data[pos : pos + matchLength]
				isValidMatch = (m1 == m2 and matchLength < self.maxStringLength)

				if isValidMatch:
					matchLength += 1
					foundMatch = True
				else:
					realMatchLength = matchLength - 1

					if foundMatch and realMatchLength > bestMatchLength:
						bestMatchDistance = pos - searchStart - realMatchLength
						bestMatchLength = realMatchLength

					matchLength = self.minStringLength
					searchStart += 1
					foundMatch = False

			if bestMatchLength:
				newCompressed = (self.referencePrefix + self.__encodeReferenceInt(bestMatchDistance, 2) + self.__encodeReferenceLength(bestMatchLength))
				pos += bestMatchLength
			else:
				if data[pos] != self.referencePrefix:
					newCompressed = data[pos]
				else:
					newCompressed = self.referencePrefix + self.referencePrefix
				pos += 1

			compressed += newCompressed

		return compressed + data[pos:].replace("`", "``")