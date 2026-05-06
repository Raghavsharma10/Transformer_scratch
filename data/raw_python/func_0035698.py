def invert(self):
        '''
        Invert by swapping each value with its key.

        Returns
        -------
        MultiDict
            Inverted multi-dict.

        Examples
        --------
        >>> MultiDict({1: {1}, 2: {1,2,3}}, 4: {}).invert()
        MultiDict({1: {1,2}, 2: {2}, 3: {2}})
        '''
        result = defaultdict(set)
        for k, val in self.items():
            result[val].add(k)
        return MultiDict(dict(result))