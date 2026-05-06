def delete_subject(self, subject_id):
        """
        Remove a specific subject by its identifier.
        """
        # subject_id could be a path such as '/role/analyst' so quote
        uri = self._get_subject_uri(guid=subject_id)
        return self.service._delete(uri)