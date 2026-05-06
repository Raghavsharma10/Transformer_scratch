def selenium(self):
        """Get the instance of webdriver, it starts the browser if the
        webdriver is not yet instantied

        :return: a `selenium instance <http://selenium-python.readthedocs.org/
        api.html#module-selenium.webdriver.remote.webdriver>`
        """
        if not self._web_driver:
            self._web_driver = self._start_driver()
        return self._web_driver