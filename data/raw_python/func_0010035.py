def close_driver(self):
        """
        Close current running instance of Webdriver.

        Usage::

            driver = WTF_WEBDRIVER_MANAGER.new_driver()
            driver.get("http://the-internet.herokuapp.com")
            WTF_WEBDRIVER_MANAGER.close_driver()
        """
        channel = self.__get_channel()
        driver = self.__get_driver_for_channel(channel)
        if self.__config.get(self.REUSE_BROWSER, True):
            # If reuse browser is set, we'll avoid closing it and just clear out the cookies,
            # and reset the location.
            try:
                driver.delete_all_cookies()
                # check to see if webdriver is still responding
                driver.get("about:blank")
            except:
                pass

        if driver is not None:
            try:
                driver.quit()
            except:
                pass

            self.__unregister_driver(channel)
            if driver in self.__registered_drivers[channel]:
                self.__registered_drivers[channel].remove(driver)

            self.webdriver = None