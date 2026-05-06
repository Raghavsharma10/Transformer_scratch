def check_css_selectors(webdriver, *selectors):
        """Returns true if all CSS selectors passed in is found.  This can be used 
        to quickly validate a page.

        Args:
            webdriver (Webdriver) : Selenium Webdriver instance
            selectors (str) : N number of CSS selectors strings to match against the page.

        Returns:
            True, False - if the page matches all selectors.

        Usage Example::

            # Checks for a Form with id='loginForm' and a button with class 'login'
            if not PageObjectUtils.check_css_selectors("form#loginForm", "button.login"):
                raise InvalidPageError("This is not the login page.")

        You can use this within a PageObject's `_validate_page(webdriver)` method for 
        validating pages.
        """
        for selector in selectors:
            try:
                webdriver.find_element_by_css_selector(selector)
            except:
                return False  # A selector failed.

        return True