def filter(self, **filters):
        """
        Add a filter to this query.
        Appends to any previous filters set.

        :rtype: Query
        """

        q = self._clone()
        for key, value in filters.items():
            filter_key = re.split('__', key)
            filter_attr = filter_key[0]
            if filter_attr not in self._valid_filter_attrs:
                raise ClientValidationError("Invalid filter attribute: %s" % key)

            # we use __ as a separator in the Python library, the APIs use '.'
            q._filters['.'.join(filter_key)].append(value)
        return q