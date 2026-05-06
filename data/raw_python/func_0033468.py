def _build_search_values(self, kwargs):
        """Build the search criteria dictionary. It will first try and build
        the values from already set attributes on the object, falling back
        to the passed in kwargs.

        :param dict kwargs: Values to build the dict from
        :rtype: dict

        """
        criteria = {}
        for key in self._search_by:
            if getattr(self, key, None):
                criteria[key] = getattr(self, key)
            elif key in kwargs and kwargs.get(key):
                criteria[key] = kwargs.get(key)
        return criteria