def _validate_archive_name(self, archive_name):
        '''
        Utility function for creating and validating archive names

        Parameters
        ----------

        archive_name: str
            Name of the archive from which to create a service path

        Returns
        -------

        archive_path: str
            Internal path used by services to reference archive data
        '''
        archive_name = fs.path.normpath(archive_name)
        patterns = self.manager.required_archive_patterns

        for pattern in patterns:
            if not re.search(pattern, archive_name):
                raise ValueError(
                    "archive name does not match pattern '{}'".format(pattern))