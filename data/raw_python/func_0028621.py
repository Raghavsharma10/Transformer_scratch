def element_value_should_contain(self, locator, expected):
		"""Verifies the element identified by `locator` contains the expected value.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| expected | expected value | Slim Shady |"""

		self._info("Verifying element '%s' value contains '%s'" % (locator, expected))
		
		element = self._element_find(locator, True, True)
		value = str(element.get_attribute('value'))
		
		if expected in value:
			return
		
		else:
			raise AssertionError("Value '%s' did not appear in element '%s'. It's value was '%s'" % (expected, locator, value))