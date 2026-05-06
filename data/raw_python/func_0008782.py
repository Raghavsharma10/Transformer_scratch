def do_march(self):
        """
        March about and trace the outline of our object

        Returns
        -------
        perimeter : list
            The pixels on the perimeter of the region [[x1, y1], ...]
        """
        x, y = self.find_start_point()
        perimeter = self.walk_perimeter(x, y)
        return perimeter