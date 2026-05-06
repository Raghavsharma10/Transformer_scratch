def get_color_index(self, colors, refresh=False):
        """Get color index.

        Refresh data from Vera if refresh is True, otherwise use local cache.
        """
        if refresh:
            self.refresh_complex_value('SupportedColors')

        sup = self.get_complex_value('SupportedColors')
        if sup is None:
            return None

        sup = sup.split(',')
        if not set(colors).issubset(sup):
            return None

        return [sup.index(c) for c in colors]