def find(self, locator, find_all=False, search_object=None, force_find=False, exclude_invisible=False):
        """
        Attempts to locate an element, trying the number of times specified by the driver wrapper;
        Will throw a WebDriverWrapperException if no element is found

        @type locator:          webdriverwrapper.support.locator.Locator
        @param locator:         the locator or css string used to query the element
        @type find_all:         bool
        @param find_all:        set to True to locate all located elements as a list
        @type search_object:    webdriverwrapper.WebElementWrapper
        @param force_find:      If true will use javascript to find elements
        @type force_find:       bool
        @param search_object:   A WebDriver or WebElement object to call find_element(s)_by_xxxxx
        """
        search_object = self.driver if search_object is None else search_object
        attempts = 0

        while attempts < self.find_attempts + 1:
            if bool(force_find):
                js_locator = self.locator_handler.parse_locator(locator)

                if js_locator.By != 'css selector':
                    raise ValueError(
                        'You must use a css locator in order to force find an element; this was "{}"'.format(
                            js_locator))

                elements = self.js_executor.execute_template_and_return_result(
                    'getElementsTemplate.js', variables={'selector': js_locator.value})
            else:
                elements = self.locator_handler.find_by_locator(search_object, locator, True)

            # Save original elements found before applying filters to the list
            all_elements = elements

            # Check for only visible elements
            visible_elements = elements
            if exclude_invisible:
                visible_elements = [element for element in all_elements if element.is_displayed()]
                elements = visible_elements

            if len(elements) > 0:
                if find_all is True:
                    # return list of wrapped elements
                    for index in range(len(elements)):
                        elements[index] = WebElementWrapper.WebElementWrapper(self, locator, elements[index],
                                                                              search_object=search_object)

                    return elements

                elif find_all is False:
                    # return first element
                    return WebElementWrapper.WebElementWrapper(self, locator, elements[0], search_object=search_object)

            else:
                if attempts >= self.find_attempts:
                    if find_all is True:  # returns an empty list if finding all elements
                        return []
                    else:  # raise exception if attempting to find one element
                        error_message = "Unable to find element after {0} attempts with locator: {1}".format(
                            attempts,
                            locator
                        )

                        # Check if filters limited the results
                        if exclude_invisible and len(visible_elements) == 0 and len(all_elements) > 0:
                            error_message = "Elements found using locator {}, but none were visible".format(locator)

                        raise WebDriverWrapperException.WebDriverWrapperException(self, error_message)
                else:
                    attempts += 1