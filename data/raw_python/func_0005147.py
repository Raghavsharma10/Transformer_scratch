def tuple(self, size=1000, tree_depth=1):
        """ Creates a random #tuple

            @size: #int number of random values to include in each @tree_depth
            @tree_depth: #int dict tree dimensions size, i.e.
                1=|(value1, value2)|
                2=|((value1, value2), (value1, value2))|

            -> random #tuple
        """
        if not tree_depth:
            return self._map_type()
        return tuple(self.tuple(size, tree_depth-1) for x in range(size))