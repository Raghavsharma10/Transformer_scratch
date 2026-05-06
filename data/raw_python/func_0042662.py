def _augment_observation_files(self, e):
        """
        Augment all the file records in an event
        :internal:
        """
        e.file_records = [self._augment_file(f) for f in e.file_records]
        return e