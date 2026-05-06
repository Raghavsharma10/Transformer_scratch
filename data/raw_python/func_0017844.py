def batch_iterator(self, batch_size, shuffle=None, **kwargs):
        """
        Create an iterator that generates mini-batches extracted from
        this data source. The batches will have `batch_size` elements, with
        the exception of the final batch which  will have less if there are
        insufficient elements left to make a complete batch.

        If `shuffle` is `None` or `False` elements will be extracted in
        order. If it is a `numpy.random.RandomState`, it will be used to
        randomise the order in which elements are extracted from the data.
        If it is `True`, NumPy's default random number generator will be
        use to shuffle elements.

        If an array of indices was provided to the constructor, the subset of
        samples identified in that array is used, rather than the complete
        set of samples.

        The generated mini-batches take the form `[batch_x, batch_y, ...]`.

        Parameters
        ----------
        batch_size: int
            Mini-batch size
        shuffle: `numpy.random.RandomState` or `True` or `None`
            Used to randomise element order. If `None`, elements will be
            extracted in order. If it is a `RandomState` instance, that
            RNG will be used to shuffle elements. If it is `True`, NumPy's
            default RNG will be used.

        Returns
        -------
        iterator
            An iterator that generates items of type `[batch_x, batch_y, ...]`
            where `batch_x`, `batch_y`, etc are themselves arrays.
        """
        for batch_ndx in self.batch_indices_iterator(
                batch_size, shuffle=shuffle, **kwargs):
            yield self.samples_by_indices_nomapping(batch_ndx)