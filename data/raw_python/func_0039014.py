def search(self, **kwargs):
        """Get movies that match the search query string from the API.

        Args:
          q (optional): plain text search query; remember to URI encode
          page_limit (optional): number of search results to show per page, 
                                 default=30
          page (optional): results page number, default=1

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_path('search')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response