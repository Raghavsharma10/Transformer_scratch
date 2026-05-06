def most_similar(self,
                     items,
                     num=10,
                     batch_size=100,
                     show_progressbar=False,
                     return_names=True):
        """
        Return the num most similar items to a given list of items.

        Parameters
        ----------
        items : list of objects or a single object.
            The items to get the most similar items to.
        num : int, optional, default 10
            The number of most similar items to retrieve.
        batch_size : int, optional, default 100.
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase the speed.
        show_progressbar : bool, optional, default False
            Whether to show a progressbar.
        return_names : bool, optional, default True
            Whether to return the item names, or just the distances.

        Returns
        -------
        sim : array
            For each items in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples. If return_names is false,
            the returned list just contains distances.

        """
        # This line allows users to input single items.
        # We used to rely on string identities, but we now also allow
        # anything hashable as keys.
        # Might fail if a list of passed items is also in the vocabulary.
        # but I can't think of cases when this would happen, and what
        # user expectations are.
        try:
            if items in self.items:
                items = [items]
        except TypeError:
            pass
        x = np.stack([self.norm_vectors[self.items[x]] for x in items])

        result = self._batch(x,
                             batch_size,
                             num+1,
                             show_progressbar,
                             return_names)

        # list call consumes the generator.
        return [x[1:] for x in result]