def element_focus_should_be_set(self, locator):
		"""Verifies the element identified by `locator` has focus.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |"""

		self._info("Verifying element '%s' focus is set" % locator)
		self._check_element_focus(True, locator)