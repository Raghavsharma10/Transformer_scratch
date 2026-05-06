def __create_safari_driver(self):
        '''
        Creates an instance of Safari webdriver.
        '''
        # Check for selenium jar env file needed for safari driver.
        if not os.getenv(self.__SELENIUM_SERVER_JAR_ENV):
            # If not set, check if we have a config setting for it.
            try:
                selenium_server_path = self._config_reader.get(
                    self.SELENIUM_SERVER_LOCATION)
                self._env_vars[
                    self.__SELENIUM_SERVER_JAR_ENV] = selenium_server_path
            except KeyError:
                raise RuntimeError(u("Missing selenium server path config {0}.").format(
                    self.SELENIUM_SERVER_LOCATION))

        return webdriver.Safari()