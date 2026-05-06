def shuffled_indices_batch_iterator(self, batch_size, shuffle_rng):
        """
        Create an iterator that generates randomly shuffled mini-batches of
        sample indices. The batches will have `batch_size` elements.

        The generated mini-batches indices take the form of 1D NumPy integer
        arrays.

        Parameters
        ----------
        batch_size: int
            Mini-batch size
        shuffle_rng: a `numpy.random.RandomState` that will be used to
            randomise element order.

        Returns
        -------
        iterator
            An iterator that generates mini-batches in the form of 1D NumPy
            integer arrays.
        """
        while True:
            yield shuffle_rng.choice(self.indices, size=(batch_size,),
                                     p=self.sub_weights)