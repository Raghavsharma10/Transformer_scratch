def nearest_neighbor_threshold(self,
                                   vectors,
                                   threshold=.5,
                                   batch_size=100,
                                   show_progressbar=False,
                                   return_names=True):
        """
        Find the nearest neighbors to some arbitrary vector.

        This function is meant to be used in composition operations. The
        most_similar function can only handle items that are in vocab, and
        looks up their vector through a dictionary. Compositions, e.g.
        "King - man + woman" are necessarily not in the vocabulary.

        Parameters
        ----------
        vectors : list of arrays or numpy array
            The vectors to find the nearest neighbors to.
        threshold : float, optional, default .5
            The threshold within to retrieve items.
        batch_size : int, optional, default 100.
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase speed.
        show_progressbar : bool, optional, default False
            Whether to show a progressbar.
        return_names : bool, optional, default True
            Whether to return the item names, or just the distances.

        Returns
        -------
        sim : list of tuples.
            For each item in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples. If return_names is set to
            false, only the distances are returned.

        """
        vectors = np.array(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        result = []

        result = self._threshold_batch(vectors,
                                       batch_size,
                                       threshold,
                                       show_progressbar,
                                       return_names)

        return list(result)