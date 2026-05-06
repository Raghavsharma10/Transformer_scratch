def movies_in_theaters(self, **kwargs):
        """Gets the movies currently in theaters from the API.

        Args:
          page_limit (optional): number of movies to show per page, default=16
          page (optional): results page number, default=1
          country (optional): localized data for selected country, default="us"

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_path('movies_in_theaters')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response