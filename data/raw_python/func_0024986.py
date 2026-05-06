def _put_subject(self, subject_id, body):
        """
        Update a subject for the given subject id.  The body is not
        a list but a dictionary of a single resource.
        """
        assert isinstance(body, (dict)), "PUT requires body to be dict."

        # subject_id could be a path such as '/asset/123' so quote
        uri = self._get_subject_uri(guid=subject_id)
        return self.service._put(uri, body)