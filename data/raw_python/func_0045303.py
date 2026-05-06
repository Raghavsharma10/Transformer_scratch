def set_data(self, data, from_db=False):
        """
        Fills the object's fields with given data dict.
        Internally calls the self._load_data() method.

        Args:
            data (dict): Data to fill object's fields.
            from_db (bool): if data coming from db then we will
            use related field type's _load_data method

        Returns:
            Self. Returns objects itself for chainability.
        """
        self._load_data(data, from_db)
        return self