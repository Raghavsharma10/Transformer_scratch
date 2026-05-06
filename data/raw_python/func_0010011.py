def get_base_url(webdriver):
        """
        Get the current base URL.

        Args:
            webdriver: Selenium WebDriver instance.

        Returns:
            str - base URL. 

        Usage::

            driver.get("http://www.google.com/?q=hello+world")
            WebUtils.get_base_url(driver)
            #returns 'http://www.google.com'

        """
        current_url = webdriver.current_url
        try:
            return re.findall("^[^/]+//[^/$]+", current_url)[0]
        except:
            raise RuntimeError(
                u("Unable to process base url: {0}").format(current_url))