def require_data(self):
        """
        raise a DatacatsError if the datadir or volumes are missing or damaged
        """
        files = task.source_missing(self.target)
        if files:
            raise DatacatsError('Missing files in source directory:\n' +
                                '\n'.join(files))
        if not self.data_exists():
            raise DatacatsError('Environment datadir missing. '
                                'Try "datacats init".')
        if not self.data_complete():
            raise DatacatsError('Environment datadir damaged or volumes '
                                'missing. '
                                'To reset and discard all data use '
                                '"datacats reset"')