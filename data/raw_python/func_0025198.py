def calc_shape_statistics(self, stat_names):
        """
        Calculate shape statistics using regionprops applied to the object mask.
        
        Args:
            stat_names: List of statistics to be extracted from those calculated by regionprops.

        Returns:
            Dictionary of shape statistics
        """
        stats = {}
        try:
            all_props = [regionprops(m) for m in self.masks]
        except TypeError:
            print(self.masks)
            exit()
        for stat in stat_names:
            stats[stat] = np.mean([p[0][stat] for p in all_props])
        return stats