def dvds_current_releases(self, **kwargs):
        """Gets the upcoming movies from the API.

        Args:
          page_limit (optional): number of movies to show per page, default=16
          page (optional): results page number, default=1
          country (optional): localized data for selected country, default="us"

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_path('dvds_current_releases')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response