def calc_attribute_statistics(self, statistic_name):
        """
        Calculates summary statistics over the domains of each attribute.
        
        Args:
            statistic_name (string): numpy statistic, such as mean, std, max, min

        Returns:
            dict of statistics from each attribute grid.
        """
        stats = {}
        for var, grids in self.attributes.items():
            if len(grids) > 1:
                stats[var] = getattr(np.array([getattr(np.ma.array(x, mask=self.masks[t] == 0), statistic_name)()
                                               for t, x in enumerate(grids)]), statistic_name)()
            else:
                stats[var] = getattr(np.ma.array(grids[0], mask=self.masks[0] == 0), statistic_name)()
        return stats