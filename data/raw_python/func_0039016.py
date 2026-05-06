def clips(self, **kwargs):
        """Get related clips and trailers for a movie specified by id 
           from the API.

        Returns:
          A dict respresentation of the JSON returned from the API.
        """
        path = self._get_id_path('clips')

        response = self._GET(path, kwargs)
        self._set_attrs_to_values(response)
        return response