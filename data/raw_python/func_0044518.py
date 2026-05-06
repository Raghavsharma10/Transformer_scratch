def addFilter(self, field, value):
        """Add a filter to the seach.

        :param field: what field filter (see GitHub search).
        :type field: str.
        :param value: value of the filter (see GitHub search).
        :type value: str.
        """
        if "<" not in value or ">" not in value or ".." not in value:
            value = ":" + value

        if self.__urlFilters:
            self.__urlFilters += "+" + field + str(quote(value))
        else:
            self.__urlFilters += field + str(quote(value))