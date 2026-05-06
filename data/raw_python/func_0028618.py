def element_height_should_be(self, locator, expected):
		"""Verifies the element identified by `locator` has the expected
		height. Expected height should be in pixels.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| expected | expected height | 600 |"""

		self._info("Verifying element '%s' height is '%s'" % (locator, expected))
		self._check_element_size(locator, 'height', expected)