def sample_ids_in_class(self, class_id):
        """
        Returns a list of sample ids belonging to a given class.

        Parameters
        ----------
        class_id : str
            class id to query.

        Returns
        -------
        subset_ids : list
            List of sample ids belonging to a given class.

        """

        # subset_ids = [sid for sid in self.keys if self.classes[sid] == class_id]
        subset_ids = self.keys_with_value(self.classes, class_id)
        return subset_ids