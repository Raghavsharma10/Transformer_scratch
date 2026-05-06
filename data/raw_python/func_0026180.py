def as_list(self, key):
        """
        A convenience method which fetches the specified value, guaranteeing
        that it is a list.

        >>> a = ConfigObj()
        >>> a['a'] = 1
        >>> a.as_list('a')
        [1]
        >>> a['a'] = (1,)
        >>> a.as_list('a')
        [1]
        >>> a['a'] = [1]
        >>> a.as_list('a')
        [1]
        """
        result = self[key]
        if isinstance(result, (tuple, list)):
            return list(result)
        return [result]