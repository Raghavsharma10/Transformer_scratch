def element_value_should_not_contain(self, locator, value):
		"""Verifies the element identified by `locator` does not contain the specified value.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| value | value it should not contain | Slim Shady |"""

		self._info("Verifying element '%s' value does not contain '%s'" % (locator, value))
		
		element = self._element_find(locator, True, True)
		elem_value = str(element.get_attribute('value'))
		
		if value in elem_value:
			raise AssertionError("Value '%s' was found in element '%s' while it shouldn't have" % (value, locator))