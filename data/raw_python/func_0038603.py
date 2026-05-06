def xpath_pick_one(self, xpaths):
		"""
		Try each of the xpaths successively until
		a single element is found. If no xpath succeeds
		then raise the last UnexpectedContentException caught.
		"""
		for xpathi, xpath in enumerate(xpaths):
			try:
				return self.xpath(xpath, [1, 1])[0]
			except UnexpectedContentException as e:
				if xpathi == len(xpaths) - 1:
					raise