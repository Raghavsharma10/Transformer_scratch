def new_driver(self, testname=None):
        '''
        Used at a start of a test to get a new instance of WebDriver.  If the 
        'resuebrowser' setting is true, it will use a recycled WebDriver instance
        with delete_all_cookies() called.

        Kwargs:
            testname (str) - Optional test name to pass to Selenium Grid.  Helpful for 
                             labeling tests on 3rd party WebDriver cloud providers.

        Returns:
            Webdriver - Selenium Webdriver instance.

        Usage::

            driver = WTF_WEBDRIVER_MANAGER.new_driver()
            driver.get("http://the-internet.herokuapp.com")
        '''
        channel = self.__get_channel()

        # Get reference for the current driver.
        driver = self.__get_driver_for_channel(channel)

        if self.__config.get(WebDriverManager.REUSE_BROWSER, True):

            if driver is None:
                driver = self._webdriver_factory.create_webdriver(
                    testname=testname)

                # Register webdriver so it can be retrieved by the manager and
                # cleaned up after exit.
                self.__register_driver(channel, driver)
            else:
                try:
                    # Attempt to get the browser to a pristine state as possible when we are
                    # reusing this for another test.
                    driver.delete_all_cookies()
                    # check to see if webdriver is still responding
                    driver.get("about:blank")
                except:
                    # In the case the browser is unhealthy, we should kill it
                    # and serve a new one.
                    try:
                        if driver.is_online():
                            driver.quit()
                    except:
                        pass

                    driver = self._webdriver_factory.create_webdriver(
                        testname=testname)
                    self.__register_driver(channel, driver)

        else:
            # Attempt to tear down any existing webdriver.
            if driver is not None:
                try:
                    driver.quit()
                except:
                    pass
            self.__unregister_driver(channel)
            driver = self._webdriver_factory.create_webdriver(
                testname=testname)
            self.__register_driver(channel, driver)

        return driver