def dvds_top_rentals(self, **kwargs):
        """Gets the current opening movies from the API.

        Args:
          limit (optional): limits the number of movies returned, default=10
          country (optional): localized data for selected country, default="us"

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_path('dvds_top_rentals')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response