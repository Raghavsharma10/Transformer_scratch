def bow(self, tokens, remove_oov=False):
        """
        Create a bow representation of a list of tokens.

        Parameters
        ----------
        tokens : list.
            The list of items to change into a bag of words representation.
        remove_oov : bool.
            Whether to remove OOV items from the input.
            If this is True, the length of the returned BOW representation
            might not be the length of the original representation.

        Returns
        -------
        bow : generator
            A BOW representation of the list of items.

        """
        if remove_oov:
            tokens = [x for x in tokens if x in self.items]

        for t in tokens:
            try:
                yield self.items[t]
            except KeyError:
                if self.unk_index is None:
                    raise ValueError("You supplied OOV items but didn't "
                                     "provide the index of the replacement "
                                     "glyph. Either set remove_oov to True, "
                                     "or set unk_index to the index of the "
                                     "item which replaces any OOV items.")
                yield self.unk_index