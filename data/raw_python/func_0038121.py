def is_attribute_visible(self, key):
        """
        Returns True if an attribute is visible
        If attribute is an instance of AttributeFilter, it returns True if all attributes
        of the sub filter are visible.

        :param key: name of attribute to check
        :type key: str
        :return: whether attribute is visible
        :rtype: bool
        """
        if key in self:
            attribute_status = getattr(self, key)
            if isinstance(attribute_status, bool) and attribute_status is True:
                return True
            elif isinstance(attribute_status, self.__class__) and attribute_status.are_any_attributes_visible():
                return True

        return False