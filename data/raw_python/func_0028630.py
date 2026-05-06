def wait_until_element_is_clickable(self, locator, timeout=None):
		"""Clicks the element specified by `locator` until the operation succeeds. This should be
		used with buttons that are generated in real-time and that don't have their click handling available
		immediately. This keyword avoids unclickable element exceptions.

		| =Argument= | =Description= | =Example= |
		| locator | Selenium 2 element locator(s) | id=MyId |
		| timeout | maximum time to wait, if set to ${None} it will use Selenium's default timeout | 5s |"""

		self._wait_until_no_error(timeout, self._wait_for_click_to_succeed, locator)