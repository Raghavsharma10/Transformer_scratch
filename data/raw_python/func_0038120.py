def is_filter_at_key(self, key):
        """
        return True if attribute is a sub filter
        """

        if key in self:
            attribute_status = getattr(self, key)
            if isinstance(attribute_status, self.__class__):
                return True

        return False