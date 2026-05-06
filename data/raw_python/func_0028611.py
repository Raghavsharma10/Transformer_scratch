def wait_until_element_value_is(self, locator, expected, strip=False, timeout=None):
		"""Waits until the element identified by `locator` value is exactly the
		expected value. You might want to use `Element Value Should Be` instead.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| expected | expected value | My Name Is Slim Shady |
		| strip | boolean, determines whether it should strip the value of the field before comparison | ${True} / ${False} |
		| timeout | maximum time to wait before the function throws an element not found error (default=None) | 5s |"""

		self._info("Waiting for '%s' value to be '%s'" % (locator, expected))
		self._wait_until_no_error(timeout, self._check_element_value_exp, False, locator, expected, strip, timeout)