def _get_subject_uri(self, guid=None):
        """
        Returns the full path that uniquely identifies
        the subject endpoint.
        """
        uri = self.uri + '/v1/subject'
        if guid:
            uri += '/' + urllib.quote_plus(guid)
        return uri