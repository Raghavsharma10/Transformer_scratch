def wait_until_page_does_not_contain_these_elements(self, timeout, *locators):
		"""Waits until all of the specified elements are not found on the page.

		| *Argument* | *Description* | *Example* |
		| timeout | maximum time to wait, if set to ${None} it will use Selenium's default timeout | 5s |
		| *locators | Selenium 2 element locator(s) | id=MyId |"""

		self._wait_until_no_error(timeout, self._wait_for_elements_to_go_away, locators)