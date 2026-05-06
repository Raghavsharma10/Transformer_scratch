def wait_until_page_contains_elements(self, timeout, *locators):
		"""This is a copy of `Wait Until Page Contains Element` but it allows
		multiple arguments in order to wait for more than one element.
		
		| *Argument* | *Description* | *Example* |
		| timeout | maximum time to wait, if set to ${None} it will use Selenium's default timeout | 5s |
		| *locators | Selenium 2 element locator(s) | id=MyId |"""
		
		self._wait_until_no_error(timeout, self._wait_for_elements, locators)