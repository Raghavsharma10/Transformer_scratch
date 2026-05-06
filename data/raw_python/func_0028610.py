def wait_until_element_does_not_have_focus(self, locator, timeout=None):
		"""Waits until the element identified by `locator` doesn't have focus.
		You might rather want to use `Element Focus Should Not Be Set`

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| timeout | maximum time to wait before the function throws an element not found error (default=None) | 5s |"""

		self._info("Waiting until '%s' does not have focus" % (locator))
		self._wait_until_no_error(timeout, self._check_element_focus_exp, False, locator, timeout)