def cast(self, **kwargs):
        """Get the cast for a movie specified by id from the API.

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_id_path('cast')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response