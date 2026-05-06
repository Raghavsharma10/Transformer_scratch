def get_driver(self):
        '''
        Get an already running instance of Webdriver. If there is none, it will create one.

        Returns:
            Webdriver - Selenium Webdriver instance.

        Usage::

            driver = WTF_WEBDRIVER_MANAGER.new_driver()
            driver.get("http://the-internet.herokuapp.com")
            same_driver = WTF_WEBDRIVER_MANAGER.get_driver()
            print(driver is same_driver) # True
        '''
        driver = self.__get_driver_for_channel(self.__get_channel())
        if driver is None:
            driver = self.new_driver()

        return driver