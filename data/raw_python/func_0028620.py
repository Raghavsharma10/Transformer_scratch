def element_value_should_not_be(self, locator, value, strip=False):
		"""Verifies the element identified by `locator` is not the specified value.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| value | value it should not be | My Name Is Slim Shady |
		| strip | Boolean, determines whether it should strip the field's value before comparison or not | ${True} / ${False} |"""

		self._info("Verifying element '%s' value is not '%s'" % (locator, value))
		
		element = self._element_find(locator, True, True)
		elem_value = str(element.get_attribute('value'))

		if (strip):
			elem_value = elem_value.strip()
		
		if elem_value == value:
			raise AssertionError("Value was '%s' for element '%s' while it shouldn't have" % (elem_value, locator))