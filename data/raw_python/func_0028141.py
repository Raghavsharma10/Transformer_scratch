def set_requestable(self, requestable=True):
        # type: (bool) -> None
        """Set the dataset to be of type requestable or not

        Args:
            requestable (bool): Set whether dataset is requestable. Defaults to True.

        Returns:
            None
        """
        self.data['is_requestdata_type'] = requestable
        if requestable:
            self.data['private'] = False