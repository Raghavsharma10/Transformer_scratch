def movie_lists(self, **kwargs):
        """Gets the movie lists available from the API.

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_path('movie_lists')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response