def element_value_should_be(self, locator, expected, strip=False):
		"""Verifies the element identified by `locator` has the expected value.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| expected | expected value | My Name Is Slim Shady |
		| strip | Boolean, determines whether it should strip the field's value before comparison or not | ${True} / ${False} |"""

		self._info("Verifying element '%s' value is '%s'" % (locator, expected))
		
		element = self._element_find(locator, True, True)
		value = element.get_attribute('value')

		if (strip):
			value = value.strip()
		
		if str(value) == expected:
			return
		
		else:
			raise AssertionError("Element '%s' value was not '%s', it was '%s'" % (locator, expected, value))