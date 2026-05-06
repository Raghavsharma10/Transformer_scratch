def invalidate(self):
        """ Rests all keys states. """
        for row in self.rows:
            for key in row.keys:
                key.state = 0