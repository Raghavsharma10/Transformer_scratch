def find_by_locator(webdriver_or_element, locator, find_all_elements=False):
        """
        Locate an element using either a webdriver or webelement

        @param webdriver_or_element:    Webdriver or Webelement object used for search
        @type locator:                  webdriver.webdriverwrapper.support.locator.Locator
        @param locator:                 locator used in search
        @type find_all_elements:        bool
        @param find_all_elements:       return all elements matching if true, first matching if false

        @return:                        either a single WebElement or a list of WebElements

        """
        # handle backwards compatibility to support new Locator class
        if isinstance(locator, loc.Locator):
            locator = '{by}={locator}'.format(by=locator.by, locator=locator.locator)

        # use the appropriate find method given the locator type;
        # locators should follow the convention "css=.class" or "xpath=//div"
        # if locator type is unspecified, it will default to css
        if (locator.count('css=') > 0  or locator.count('css_selector=')) and len(locator.split('=', 1)) > 1:
            if find_all_elements:
                return webdriver_or_element.find_elements_by_css_selector(locator.split('=', 1)[-1])
            else:
                return webdriver_or_element.find_element_by_css_selector(locator.split('=', 1)[-1])

        elif locator.count('id=') > 0 and len(locator.split('=')) > 1:
            if find_all_elements:
                return webdriver_or_element.find_elements_by_id(locator.split('=', 1)[-1])
            else:
                return webdriver_or_element.find_element_by_id(locator.split('=', 1)[-1])

        elif locator.count('xpath=') > 0 and len(locator.split('=')) > 1:
            if find_all_elements:
                return webdriver_or_element.find_elements_by_xpath(locator.split('=', 1)[-1])
            else:
                return webdriver_or_element.find_element_by_xpath(locator.split('=', 1)[-1])

        elif locator.count('class_name=') > 0 and len(locator.split('=')) > 1:
            if find_all_elements:
                return webdriver_or_element.find_elements_by_class_name(locator.split('=', 1)[-1])
            else:
                return webdriver_or_element.find_element_by_class_name(locator.split('=', 1)[-1])

        elif locator.count('link_text=') > 0 and len(locator.split('=')) > 1:
            if find_all_elements:
                return webdriver_or_element.find_elements_by_link_text(locator.split('=', 1)[-1])
            else:
                return webdriver_or_element.find_element_by_link_text(locator.split('=', 1)[-1])

        elif locator.count('partial_link_text=') > 0 and len(locator.split('=')) > 1:
            if find_all_elements:
                return webdriver_or_element.find_elements_by_partial_link_text(locator.split('=', 1)[-1])
            else:
                return webdriver_or_element.find_element_by_partial_link_text(locator.split('=', 1)[-1])

        elif locator.count('name=') > 0 and len(locator.split('=')) > 1:
            if find_all_elements:
                return webdriver_or_element.find_elements_by_name(locator.split('=', 1)[-1])
            else:
                return webdriver_or_element.find_element_by_name(locator.split('=', 1)[-1])

        elif locator.count('tag_name=') > 0 and len(locator.split('=')) > 1:
            if find_all_elements:
                return webdriver_or_element.find_elements_by_tag_name(locator.split('=', 1)[-1])
            else:
                return webdriver_or_element.find_element_by_tag_name(locator.split('=', 1)[-1])

        else:   # default to css
            if find_all_elements:
                return webdriver_or_element.find_elements_by_css_selector(locator)
            else:
                return webdriver_or_element.find_element_by_css_selector(locator)