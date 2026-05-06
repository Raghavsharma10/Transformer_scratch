def create_page(cls, webdriver=None, **kwargs):
        """Class method short cut to call PageFactory on itself.  Use it to instantiate 
        this PageObject using a webdriver.

        Args:
            webdriver (Webdriver): Instance of Selenium Webdriver.

        Returns:
            PageObject

        Raises:
            InvalidPageError

        """
        if not webdriver:
            webdriver = WTF_WEBDRIVER_MANAGER.get_driver()
        return PageFactory.create_page(cls, webdriver=webdriver, **kwargs)