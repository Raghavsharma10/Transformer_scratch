def in_order_indices_batch_iterator(self, batch_size):
        """
        Create an iterator that generates in-order mini-batches of sample
        indices. The batches will have `batch_size` elements, with the
        exception of the final batch which will have less if there are not
        enough samples left to fill it.

        The generated mini-batches indices take the form of 1D NumPy integer
        arrays.

        Parameters
        ----------
        batch_size: int
            Mini-batch size

        Returns
        -------
        iterator
            An iterator that generates mini-batches in the form of 1D NumPy
            integer arrays.
        """
        if self.repeats == 1:
            for i in range(0, self.length, batch_size):
                yield np.arange(i, min(i + batch_size, self.length))
        else:
            repeats = self.repeats
            i = 0
            while True:
                j = i + batch_size
                if j <= self.length:
                    # Within size of data
                    yield np.arange(i, j)
                    i = j
                elif j <= self.length * 2:
                    # One restart is required
                    # Reduce the number of remaining repeats
                    if repeats != -1:
                        repeats -= 1
                    if repeats == 0:
                        # Finished; emit remaining elements
                        if i < self.length:
                            yield np.arange(i, self.length)
                        break

                    # Wrap over
                    # Compute number of elements required to make up
                    # the batch
                    k = batch_size - (self.length - i)
                    yield np.append(np.arange(i, self.length),
                                    np.arange(0, k), axis=0)
                    i = k
                else:
                    # Multiple restarts required to fill the batch
                    batch_ndx = np.arange(0)
                    # i = 0
                    while len(batch_ndx) < batch_size:
                        # Wrap over
                        k = min(batch_size - len(batch_ndx), self.length - i)
                        batch_ndx = np.append(
                            batch_ndx, np.arange(i, i + k), axis=0)
                        i += k
                        if i >= self.length:
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