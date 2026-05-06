def get_subject(self, subject_id):
        """
        Returns a specific subject by subject id.
        """
        # subject_id could be a path such as '/user/j12y' so quote
        uri = self._get_subject_uri(guid=subject_id)
        return self.service._get(uri)