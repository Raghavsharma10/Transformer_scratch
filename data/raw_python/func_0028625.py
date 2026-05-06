def element_css_attribute_should_be(self, locator, prop, expected):
		"""Verifies the element identified by `locator` has the expected
		value for the targeted `prop`.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| prop | targeted css attribute | background-color |
		| expected | expected value | rgba(0, 128, 0, 1) |"""

		self._info("Verifying element '%s' has css attribute '%s' with a value of '%s'" % (locator, prop, expected))
		self._check_element_css_value(locator, prop, expected)