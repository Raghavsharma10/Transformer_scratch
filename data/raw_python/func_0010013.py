def is_webdriver_mobile(webdriver):
        """
        Check if a web driver if mobile.

        Args:
            webdriver (WebDriver): Selenium webdriver.

        """
        browser = webdriver.capabilities['browserName']

        if (browser == u('iPhone') or 
            browser == u('android')):
            return True
        else:
            return False