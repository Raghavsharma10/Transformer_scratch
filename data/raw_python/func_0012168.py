def vectorize(self, tokens, remove_oov=False, norm=False):
        """
        Vectorize a sentence by replacing all items with their vectors.

        Parameters
        ----------
        tokens : object or list of objects
            The tokens to vectorize.
        remove_oov : bool, optional, default False
            Whether to remove OOV items. If False, OOV items are replaced by
            the UNK glyph. If this is True, the returned sequence might
            have a different length than the original sequence.
        norm : bool, optional, default False
            Whether to return the unit vectors, or the regular vectors.

        Returns
        -------
        s : numpy array
            An M * N matrix, where every item has been replaced by
            its vector. OOV items are either removed, or replaced
            by the value of the UNK glyph.

        """
        if not tokens:
            raise ValueError("You supplied an empty list.")
        index = list(self.bow(tokens, remove_oov=remove_oov))
        if not index:
            raise ValueError("You supplied a list with only OOV tokens: {}, "
                             "which then got removed. Set remove_oov to False,"
                             " or filter your sentences to remove any in which"
                             " all items are OOV.")
        if norm:
            return np.stack([self.norm_vectors[x] for x in index])
        else:
            return np.stack([self.vectors[x] for x in index])