def find(self, upload_id, **kwargs):
        """
        Finds an upload by ID.
        """

        return super(UploadsProxy, self).find(upload_id, file_upload=True)