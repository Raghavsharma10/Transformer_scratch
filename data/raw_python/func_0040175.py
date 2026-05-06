def isObservableElement(self, elementName):
        """
        Mention if an element is an observable element.

        :param str ElementName: the element name to evaluate
        :return: true if is an observable element, otherwise false.
        :rtype: bool
        """
        if not(isinstance(elementName, str)):
            raise TypeError(
                "Element name should be a string ." +
                "I receive this {0}"
                .format(elementName))

        return (True if (elementName == "*")
                else self._evaluateString(elementName))