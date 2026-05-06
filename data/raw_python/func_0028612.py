def wait_until_element_value_contains(self, locator, expected, timeout=None):
		"""Waits until the element identified by `locator` contains
		the expected value. You might want to use `Element Value Should Contain` instead.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| expected | expected value | Slim Shady |
		| timeout | maximum time to wait before the function throws an element not found error (default=None) | 5s |"""

		self._info("Waiting for '%s' value to contain '%s'" % (locator, expected))
		self._wait_until_no_error(timeout, self._check_element_value_exp, True, locator, expected, False, timeout)