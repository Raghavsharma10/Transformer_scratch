def element_width_should_be(self, locator, expected):
		"""Verifies the element identified by `locator` has the expected
		width. Expected width should be in pixels.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| expected | expected width | 800 |"""

		self._info("Verifying element '%s' width is '%s'" % (locator, expected))
		self._check_element_size(locator, 'width', expected)