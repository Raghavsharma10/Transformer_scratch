def get_for_model(self, ModelClass):
        """
        Return the URL type for a given model class
        """
        for urltype in self._url_types:
            if urltype.model is ModelClass:
                return urltype
        return None