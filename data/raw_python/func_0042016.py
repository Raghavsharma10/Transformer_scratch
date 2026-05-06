def __mark(self, element, mark_set):
        """
        Marks an element

        :param element: The element to mark
        :param mark_set: The set corresponding to the mark
        :return: True if the element was known
        """
        try:
            # The given element can be of a different type than the original
            # one (JID instead of str, ...), so we retrieve the original one
            original = self.__elements.pop(element)
            mark_set.add(original)
        except KeyError:
            return False
        else:
            if not self.__elements:
                # No more elements to wait for
                self.__call()
            return True