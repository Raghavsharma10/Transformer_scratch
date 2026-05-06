def box_coordinates(self):
        """Returns a thumbnail's coordinates."""
        if (
            self.thumb_x is not None and
            self.thumb_y is not None and
            self.thumb_x2 is not None and
            self.thumb_y2 is not None
        ):
            return (
                int(self.thumb_x),
                int(self.thumb_y),
                int(self.thumb_x2),
                int(self.thumb_y2),
            )
        return False