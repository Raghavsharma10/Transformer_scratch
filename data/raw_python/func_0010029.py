def create_webdriver(self, testname=None):
        '''
            Creates an instance of Selenium webdriver based on config settings.
            This should only be called by a shutdown hook.  Do not call directly within 
            a test.

            Kwargs:
                testname: Optional test name to pass, this gets appended to the test name 
                          sent to selenium grid.

            Returns:
                WebDriver - Selenium Webdriver instance.

        '''
        try:
            driver_type = self._config_reader.get(
                self.DRIVER_TYPE_CONFIG)
        except:
            driver_type = self.DRIVER_TYPE_LOCAL
            _wtflog.warn("%s setting is missing from config. Using default setting, %s",
                         self.DRIVER_TYPE_CONFIG, driver_type)

        if driver_type == self.DRIVER_TYPE_REMOTE:
            # Create desired capabilities.
            self.webdriver = self.__create_remote_webdriver_from_config(
                testname=testname)
        else:
            # handle as local webdriver
            self.webdriver = self.__create_driver_from_browser_config()
        try:
            self.webdriver.maximize_window()
        except:
            # wait a short period and try again.
            time.sleep(self._timeout_mgr.BRIEF)
            try:
                self.webdriver.maximize_window()
            except Exception as e:
                if (isinstance(e, WebDriverException) and
                    "implemented" in e.msg.lower()):
                    pass  # Maximizing window not supported by this webdriver.
                else:
                    _wtflog.warn("Unable to maxmize browser window. " + 
                                 "It may be possible the browser did not instantiate correctly. % s",
                                 e)

        return self.webdriver