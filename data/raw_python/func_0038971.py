def get_class(self, class_id):
        """
        Returns a smaller dataset belonging to the requested classes.

        Parameters
        ----------
        class_id : str or list
            identifier(s) of the class(es) to be returned.

        Returns
        -------
        MLDataset
            With subset of samples belonging to the given class(es).

        Raises
        ------
        ValueError
            If one or more of the requested classes do not exist in this dataset.
            If the specified id is empty or None

        """
        if class_id in [None, '']:
            raise ValueError("class id can not be empty or None.")

        if isinstance(class_id, str):
            class_ids = [class_id, ]
        else:
            class_ids = class_id

        non_existent = set(self.class_set).intersection(set(class_ids))
        if len(non_existent) < 1:
            raise ValueError(
                'These classes {} do not exist in this dataset.'.format(non_existent))

        subsets = list()
        for class_id in class_ids:
            subsets_this_class = self.keys_with_value(self.__classes, class_id)
            subsets.extend(subsets_this_class)

        return self.get_subset(subsets)