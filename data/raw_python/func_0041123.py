def set_query(self, value):
        """ Convert a dict form of query in a string of needed and store the query string.

            Args:
                value -- A query string or a dict with query xpaths as keys and text or
                        nested query dicts as values.
        """
        if isinstance(value, basestring) or value is None:
            self._content['query'] = value
        elif hasattr(value, 'keys'):
            self._content['query'] = query.terms_from_dict(value)
        else:
            raise TypeError("Query must be a string or dict. Got: " + type(value) + " insted!")