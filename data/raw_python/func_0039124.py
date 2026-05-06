def movies_box_office(self, **kwargs):
        """Gets the top box office earning movies from the API.
           Sorted by most recent weekend gross ticket sales.

        Args:
          limit (optional): limits the number of movies returned, default=10
          country (optional): localized data for selected country, default="us"

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_path('movies_box_office')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response