def _create_archive(
            self,
            archive_name,
            metadata):
        '''
        This adds an item in a DynamoDB table corresponding to a S3 object

        Args
        ----
        arhive_name: str
            corresponds to the name of the Archive (e.g. )


        Returns
        -------
        Dictionary with confirmation of upload

        '''

        archive_exists = False

        try:
            self.get_archive(archive_name)
            archive_exists = True
        except KeyError:
            pass

        if archive_exists:
            raise KeyError(
                "{} already exists. Use get_archive() to view".format(
                    archive_name))

        self._table.put_item(Item=metadata)