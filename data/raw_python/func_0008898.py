def set_sides(self, key, data, field, local=False):
        """
        Assign data on the 'key' tile to all the edges
        """
        for side in ['left', 'right', 'top', 'bottom']:
            self.set(key, data, field, side, local)