def get_tip_coordinates(self, axis=None):
        """
        Returns coordinates of the tip positions for a tree. If no argument
        for axis then a 2-d array is returned. The first column is the x 
        coordinates the second column is the y-coordinates. If you enter an 
        argument for axis then a 1-d array will be returned of just that axis.
        """
        # get coordinates array
        coords = self.get_node_coordinates()
        if axis == 'x':
            return coords[:self.ntips, 0]
        elif axis == 'y':
            return coords[:self.ntips, 1]
        return coords[:self.ntips]