def similarity(self, i1, i2):
        """
        Compute the similarity between two sets of items.

        Parameters
        ----------
        i1 : object
            The first set of items.
        i2 : object
            The second set of item.

        Returns
        -------
        sim : array of floats
            An array of similarity scores between 1 and 0.

        """
        try:
            if i1 in self.items:
                i1 = [i1]
        except TypeError:
            pass
        try:
            if i2 in self.items:
                i2 = [i2]
        except TypeError:
            pass
        i1_vec = np.stack([self.norm_vectors[self.items[x]] for x in i1])
        i2_vec = np.stack([self.norm_vectors[self.items[x]] for x in i2])
        return i1_vec.dot(i2_vec.T)