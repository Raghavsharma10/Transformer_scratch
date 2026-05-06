def order_by(self, sort_key=None):
        """
        Set the sort for this query. Not all attributes are sorting candidates.
        To sort in descending order, call ``Query.order_by('-attribute')``.

        Calling ``Query.order_by()`` replaces any previous ordering.

        :rtype: Query
        """
        if sort_key is not None:
            sort_attr = re.match(r'(-)?(.*)$', sort_key).group(2)
            if sort_attr not in self._valid_sort_attrs:
                raise ClientValidationError("Invalid ordering attribute: %s" % sort_key)

        q = self._clone()
        q._order_by = sort_key
        return q