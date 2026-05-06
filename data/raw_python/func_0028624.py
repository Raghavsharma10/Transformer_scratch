def element_focus_should_not_be_set(self, locator):
		"""Verifies the element identified by `locator` does not have focus.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |"""

		self._info("Verifying element '%s' focus is not set" % locator)
		self._check_element_focus(False, locator)