def is_webdriver_ios(webdriver):
        """
        Check if a web driver if mobile.

        Args:
            webdriver (WebDriver): Selenium webdriver.

        """
        browser = webdriver.capabilities['browserName']

        if (browser == u('iPhone') or 
            browser == u('iPad')):
            return True
        else:
            return False