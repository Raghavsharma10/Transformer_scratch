def get_id(self, natural_key, enhancement=None):
        """
        Returns the technical ID for a natural key or None if the given natural key is not valid.

        :param T natural_key: The natural key.
        :param T enhancement: Enhancement data of the dimension row.

        :rtype: int|None
        """
        # If the natural key is known return the technical ID immediately.
        if natural_key in self._map:
            return self._map[natural_key]

        # The natural key is not in the map of this dimension. Call a stored procedure for translating the natural key
        # to a technical key.
        self.pre_call_stored_procedure()
        success = False
        try:
            key = self.call_stored_procedure(natural_key, enhancement)
            success = True
        finally:
            self.post_call_stored_procedure(success)

        # Add the translation for natural key to technical ID to the map.
        self._map[natural_key] = key

        return key