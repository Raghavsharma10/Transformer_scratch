def _dataset_load_from_hdx(self, id_or_name):
        # type: (str) -> bool
        """Loads the dataset given by either id or name from HDX

        Args:
            id_or_name (str): Either id or name of dataset

        Returns:
            bool: True if loaded, False if not
        """

        if not self._load_from_hdx('dataset', id_or_name):
            return False
        self._dataset_create_resources()
        return True