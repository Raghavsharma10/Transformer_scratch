def _init_randgen(self):
        """
        Initialise random generator to be used for picking elements.
        With the current implementation in tohu (where we pick elements
        from generators individually instead of in bulk), it is faster
        to `use random.Random` than `numpy.random.RandomState` (it is
        possible that this may change in the future if we change the
        design so that tohu pre-produces elements in bulk, but that's
        not likely to happen in the near future).

        Since `random.Random` doesn't support arbitrary distributions,
        we can only use it if `p=None`. This helper function returns
        the appropriate random number generator depending in the value
        of `p`, and also returns a function `random_choice` which can be
        applied to the input sequence to select random elements from it.
        """
        if self.p is None:
            self.randgen = Random()
            self.func_random_choice = partial(self.randgen.choices, k=self.num)
        else:
            self.randgen = np.random.RandomState()
            self.func_random_choice = partial(self.randgen.choice, p=self.p, k=self.num)