def find_element_by_selectors(webdriver, *selectors):
        """
        Utility method makes it easier to find an element using multiple selectors. This is 
        useful for problematic elements what might works with one browser, but fail in another.
        (Like different page elements being served up for different browsers)

        Args:
            selectors - var arg if N number of selectors to match against.  Each selector should 
                        be a Selenium 'By' object.

        Usage::
            my_element = WebElementSelector.find_element_by_selectors(webdriver,
                                                                    (By.ID, "MyElementID"),
                                                                    (By.CSS, "MyClassSelector") )

        """
        # perform initial check to verify selectors are valid by statements.
        for selector in selectors:
            (by_method, value) = selector
            if not WebElementSelector.__is_valid_by_type(by_method):
                raise BadSelectorError(
                    u("Selectors should be of type selenium.webdriver.common.by.By"))
            if type(value) != str:
                raise BadSelectorError(
                    u("Selectors should be of type selenium.webdriver.common.by.By"))

        selectors_used = []
        for selector in selectors:
            (by_method, value) = selector
            selectors_used.append(
                u("{by}:{value}").format(by=by_method, value=value))
            try:
                return webdriver.find_element(by=by_method, value=value)
            except:
                pass

        raise ElementNotSelectableException(
            u("Unable to find elements using:") + u(",").join(selectors_used))