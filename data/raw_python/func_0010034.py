def clean_up_webdrivers(self):
        '''
        Clean up webdrivers created during execution.
        '''
        # Quit webdrivers.
        _wtflog.info("WebdriverManager: Cleaning up webdrivers")
        try:
            if self.__use_shutdown_hook:
                for key in self.__registered_drivers.keys():
                    for driver in self.__registered_drivers[key]:
                        try:
                            _wtflog.debug(
                                "Shutdown hook closing Webdriver for thread: %s", key)
                            driver.quit()
                        except:
                            pass
        except:
            pass