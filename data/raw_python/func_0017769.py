def shuffled_indices_batch_iterator(self, batch_size, shuffle_rng):
        """
        Create an iterator that generates randomly shuffled mini-batches of
        sample indices. The batches will have `batch_size` elements, with the
        exception of the final batch which will have less if there are not
        enough samples left to fill it.

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
        if self.repeats == 1:
            indices = shuffle_rng.permutation(self.length)
            for i in range(0, self.length, batch_size):
                yield indices[i:i + batch_size]
        else:
            repeats = self.repeats
            indices = shuffle_rng.permutation(self.length)
            i = 0
            while True:
                j = i + batch_size
                if j <= self.length:
                    # Within size of data
                    yield indices[i:j]
                    i = j
                else:
                    # Multiple restarts required to fill the batch
                    batch_ndx = np.arange(0)
                    while len(batch_ndx) < batch_size:
                        # Wrap over
                        k = min(batch_size - len(batch_ndx), self.length - i)
                        batch_ndx = np.append(
                            batch_ndx, indices[i:i + k], axis=0)
                        i += k

                        if i >= self.length:
                            # Loop over; new permutation
                            indices = shuffle_rng.permutation(self.length)
                            i -= self.length
                            # Reduce the number of remaining repeats
                            if repeats != -1:
                                repeats -= 1
                            if repeats == 0:
                                break

                    if len(batch_ndx) > 0:
                        yield batch_ndx
                    if repeats == 0:
                        break