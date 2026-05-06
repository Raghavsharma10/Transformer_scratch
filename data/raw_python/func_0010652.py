def flatten(self, in_place=True):
        """ Merge all datasets into a single dataset.

        The default dataset is the last dataset to be merged, as it is considered to be
        the primary source of information and should overwrite all existing fields with
        the same key.

        Args:
            in_place (bool): Set to ``True`` to replace the existing datasets with the
                merged one. If set to ``False``, will return a new MultiTaskData
                object containing the merged dataset.

        Returns:
            MultiTaskData: If the in_place flag is set to False.
        """
        new_dataset = TaskData()

        for i, dataset in enumerate(self._datasets):
            if i != self._default_index:
                new_dataset.merge(dataset)

        new_dataset.merge(self.default_dataset)

        # point all aliases to the new, single dataset
        new_aliases = {alias: 0 for alias, _ in self._aliases.items()}

        # replace existing datasets or return a new MultiTaskData object
        if in_place:
            self._datasets = [new_dataset]
            self._aliases = new_aliases
            self._default_index = 0
        else:
            return MultiTaskData(dataset=new_dataset, aliases=list(new_aliases.keys()))