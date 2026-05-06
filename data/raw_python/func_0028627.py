def wait_until_page_contains_one_of_these_elements(self, timeout, *locators):
		"""Waits until at least one of the specified elements is found.
		
		| *Argument* | *Description* | *Example* |
		| timeout | maximum time to wait, if set to ${None} it will use Selenium's default timeout | 5s |
		| *locators | Selenium 2 element locator(s) | id=MyId |"""
		
		self._wait_until_no_error(timeout, self._wait_for_at_least_one_element, locators)