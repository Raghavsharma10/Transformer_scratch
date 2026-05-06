def fill_array(self, array, field, add=False, maximize=False):
        """
        Given a full array (for the while image), fill it with the data on
        the edges.
        """
        self.fix_shapes()
        for i in xrange(self.n_chunks):
            for side in ['left', 'right', 'top', 'bottom']:
                edge = getattr(self, side).ravel()[i]
                if add:
                    array[edge.slice] += getattr(edge, field)
                elif maximize:
                    array[edge.slice] = np.maximum(array[edge.slice],
                                                   getattr(edge, field))
                else:
                    array[edge.slice] = getattr(edge, field)
        return array