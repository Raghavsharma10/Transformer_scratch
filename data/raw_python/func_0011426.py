def _dom_class(self, obj1, obj2):
        """Return the dominating numeric class between the two

        :obj1: TODO
        :obj2: TODO
        :returns: TODO

        """
        if isinstance(obj1, Double) or isinstance(obj2, Double):
            return Double
        if isinstance(obj1, Float) or isinstance(obj2, Float):
            return Float