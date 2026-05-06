def has_class(self, classname):
        """Test if an element has a specific classname

        @type classname: str
        @param classname: Classname to test for; cannot contain spaces
        @rtype: bool
        @return: True if classname exists; false otherwise
        """
        def element_has_class():
            """Wrapper to test if element has a class"""
            pattern = re.compile('(\s|^){classname}(\s|$)'.format(classname=classname))
            classes = self.element.get_attribute('class')
            matches = re.search(pattern, classes)

            if matches is not None:
                return True
            return False

        return self.execute_and_handle_webelement_exceptions(
            element_has_class,
            'check for element class "{}"'.format(classname)
        )