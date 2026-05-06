def defaultdict(self, key_depth=1000, tree_depth=1):
        """ Creates a random :class:collections.defaultdict

            @key_depth: #int number of keys per @tree_depth to generate random
                values for
            @tree_depth: #int dict tree dimensions size, i.e.
                1=|{key: value}|
                2=|{key: {key: value}, key2: {key2: value2}}|

            -> random :class:collections.defaultdict
        """
        if not tree_depth:
            return self._map_type()
        _dict = defaultdict()
        _dict.update({
            self.randstr: self.defaultdict(key_depth, tree_depth-1)
            for x in range(key_depth)})
        return _dict