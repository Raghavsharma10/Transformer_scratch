def stepThroughJsWaf_bare_chromium(self, url, titleContains='', titleNotContains='', extra_tid=None):
		'''
		Use Chromium to access a resource behind WAF protection.

		Params:
			``url`` - The URL to access that is protected by WAF
			``titleContains`` - A string that is in the title of the protected page, and NOT the
				WAF intermediate page. The presence of this string in the page title
				is used to determine whether the WAF protection has been successfully
				penetrated.
			``titleContains`` - A string that is in the title of the WAF intermediate page
				and NOT in the target page. The presence of this string in the page title
				is used to determine whether the WAF protection has been successfully
				penetrated.

		The current WebGetRobust headers are installed into the selenium browser, which
		is then used to access the protected resource.

		Once the protected page has properly loaded, the WAF access cookie is
		then extracted from the selenium browser, and installed back into the WebGetRobust
		instance, so it can continue to use the WAF auth in normal requests.

		'''

		if (not titleContains) and (not titleNotContains):
			raise ValueError("You must pass either a string the title should contain, or a string the title shouldn't contain!")

		if titleContains and titleNotContains:
			raise ValueError("You can only pass a single conditional statement!")

		self.log.info("Attempting to access page through WAF browser verification.")

		current_title = None

		if extra_tid is True:
			extra_tid = threading.get_ident()

		with self._chrome_context(url, extra_tid=extra_tid) as cr:
			self._syncIntoChromium(cr)
			cr.blocking_navigate(url)

			for _ in range(self.wrapper_step_through_timeout):
				time.sleep(1)
				current_title, _ = cr.get_page_url_title()
				if titleContains and titleContains in current_title:
					self._syncOutOfChromium(cr)
					return True
				if titleNotContains and current_title and titleNotContains not in current_title:
					self._syncOutOfChromium(cr)
					return True

			self._syncOutOfChromium(cr)

		self.log.error("Failed to step through. Current title: '%s'", current_title)

		return False