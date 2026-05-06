def store_extra_keys(self, d: Dict[str, Any]) -> None:
        """
        Store several extra values in the messaging storage.

        :param d: dictionary entry to merge with current self.extra_keys.
        :returns: None
        """
        new_dict = dict(self.extra_keys, **d)
        self.extra_keys = new_dict.copy()