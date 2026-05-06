def areObservableElements(self, elementNames):
        """
        Mention if all elements are observable element.

        :param str ElementName: the element name to evaluate
        :return: true if is an observable element, otherwise false.
        :rtype: bool
        """
        if not(hasattr(elementNames, "__len__")):
            raise TypeError(
                "Element name should be a array of strings." +
                "I receive this {0}"
                .format(elementNames))

        return self._evaluateArray(elementNames)