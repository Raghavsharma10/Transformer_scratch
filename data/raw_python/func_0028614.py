def clear_input_field(self, locator, method=0):
		"""Clears the text field identified by `locator`

		The element.clear() method doesn't seem to work properly on
		all browsers, so this keyword was created to offer alternatives.

		The `method` argument defines the method it should use in order
		to clear the target field.

		0 = Uses the selenium method by doing element.clear \n
		1 = Sets focus on the field and presses CTRL + A, and then DELETE \n
		2 = Repeatedly presses BACKSPACE until the field is empty

		This keyword, when using a method other than '2' does not validate it
		successfully cleared the field, you should handle this verification by yourself.
		When using the method '2', it presses delete until the field's value is empty.

		| *Argument* | *Description* | *Example* |
		| locator | Selenium 2 element locator | id=my_id |
		| method | the clearing method that should be used | no example provided |"""

		element = self._element_find(locator, True, True)
		
		if (int(method) == 0):

			self._info("Clearing input on element '%s'" % (locator))
			element.clear()

		elif (int(method) == 1):

			self._info("Clearing input on element '%s' by pressing 'CTRL + A + DELETE'" % (locator))
			element.send_keys(Keys.CONTROL + 'a')
			element.send_keys(Keys.DELETE)

		elif (int(method) == 2):

			self._info("Clearing input on element '%s' by repeatedly pressing BACKSPACE" % (locator))
			while (len(element.get_attribute('value')) != 0):

				element.send_keys(Keys.BACKSPACE)

		else: element.clear()