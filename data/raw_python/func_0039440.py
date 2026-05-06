def stepThroughJsWaf_selenium_chromium(self, url, titleContains='', titleNotContains=''):
		'''
		Use Selenium+SeleniumChromium to access a resource behind cloudflare protection.

		Params:
			``url`` - The URL to access that is protected by cloudflare
			``titleContains`` - A string that is in the title of the protected page, and NOT the
				cloudflare intermediate page. The presence of this string in the page title
				is used to determine whether the cloudflare protection has been successfully
				penetrated.

		The current WebGetRobust headers are installed into the selenium browser, which
		is then used to access the protected resource.

		Once the protected page has properly loaded, the cloudflare access cookie is
		then extracted from the selenium browser, and installed back into the WebGetRobust
		instance, so it can continue to use the cloudflare auth in normal requests.

		'''

		if (not titleContains) and (not titleNotContains):
			raise ValueError("You must pass either a string the title should contain, or a string the title shouldn't contain!")

		if titleContains and titleNotContains:
			raise ValueError("You can only pass a single conditional statement!")

		self.log.info("Attempting to access page through cloudflare browser verification.")

		if not self.selenium_chromium_driver:
			self._initSeleniumChromiumWebDriver()
		self._syncIntoSeleniumChromiumWebDriver()


		self.selenium_chromium_driver.get(url)

		if titleContains:
			condition = EC.title_contains(titleContains)
		elif titleNotContains:
			condition = SeleniumCommon.title_not_contains(titleNotContains)
		else:
			raise ValueError("Wat?")


		try:
			WebDriverWait(self.selenium_chromium_driver, 45).until(condition)
			success = True
			self.log.info("Successfully accessed main page!")
		except TimeoutException:
			self.log.error("Could not pass through cloudflare blocking!")
			success = False
		# Add cookies to cookiejar

		self._syncOutOfSeleniumChromiumWebDriver()

		self._syncCookiesFromFile()

		return success