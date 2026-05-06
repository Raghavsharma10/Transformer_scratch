def set_element_focus(self, locator):
		"""Sets focus on the element identified by `locator`. Should
		be used with elements meant to have focus only, such as
		text fields. This keywords also waits for the focus to be
		active by calling the `Wait Until Element Has Focus` keyword.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |"""

		self._info("Setting focus on element '%s'" % (locator))
		
		element = self._element_find(locator, True, True)
		element.send_keys(Keys.NULL)

		self._wait_until_no_error(None, self._check_element_focus, True, locator)