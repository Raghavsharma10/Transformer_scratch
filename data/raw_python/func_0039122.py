def lists(self, **kwargs):
        """Gets the top-level lists available from the API.

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_path('lists')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response