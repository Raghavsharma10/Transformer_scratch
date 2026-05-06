def intersects(self, other_grid_coordinates):
        """ returns True if the GC's overlap. """
        ogc = other_grid_coordinates  # alias
        # for explanation: http://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other
        # Note the flipped y-coord in this coord system.
        ax1, ay1, ax2, ay2 = self.ULC.lon, self.ULC.lat, self.LRC.lon, self.LRC.lat
        bx1, by1, bx2, by2 = ogc.ULC.lon, ogc.ULC.lat, ogc.LRC.lon, ogc.LRC.lat
        if ((ax1 <= bx2) and (ax2 >= bx1) and (ay1 >= by2) and (ay2 <= by1)):
            return True
        else:
            return False