def wrap_key(self, key):
        """Translate the key into the central cell

           This method is only applicable in case of a periodic system.
        """
        return tuple(np.round(
            self.integer_cell.shortest_vector(key)
        ).astype(int))