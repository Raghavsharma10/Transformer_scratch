def find_by_dynamic_locator(self, template_locator, variables, find_all=False, search_object=None):
        '''
        Find with dynamic locator

        @type template_locator:         webdriverwrapper.support.locator.Locator
        @param template_locator:        Template locator w/ formatting bits to insert
        @type variables:                dict
        @param variables:               Dictionary of variable substitutions
        @type find_all:                 bool
        @param find_all:                True to find all elements immediately, False for find single element only
        @type search_object:            webdriverwrapper.WebElementWrapper
        @param search_object:           Optional WebElement to start search with.
                                        If null, search will be on self.driver

        @rtype:                         webdriverwrapper.WebElementWrapper or list()
        @return:                        Single WebElemetnWrapper if find_all is False,
                                        list of WebElementWrappers if find_all is True
        '''
        template_variable_character = '%'
        # raise an exception if user passed non-dictionary variables
        if not isinstance(variables, dict):
            raise TypeError('You must use a dictionary to populate locator variables')

        # replace all variables that match the keys in 'variables' dict
        locator = ""
        for key in variables.keys():
            locator = template_locator.replace(template_variable_character + key, variables[key])

        return self.find(locator, find_all, search_object)