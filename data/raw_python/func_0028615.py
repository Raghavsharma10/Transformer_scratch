def element_text_color_should_be(self, locator, expected):
		"""Verifies the element identified by `locator` has the expected
		text color (it verifies the CSS attribute color). Color should be in
		RGBA format.

		Example of rgba format: rgba(RED, GREEN, BLUE, ALPHA)

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| expected | expected color | rgba(0, 128, 0, 1) |"""

		self._info("Verifying element '%s' has text color '%s'" % (locator, expected))
		self._check_element_css_value(locator, 'color', expected)