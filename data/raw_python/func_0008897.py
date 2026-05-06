def set_i(self, i, data, field, side):
        """ Assigns data on the i'th tile to the data 'field' of the 'side'
        edge of that tile
        """
        edge = self.get_i(i, side)
        setattr(edge, field, data[edge.slice])