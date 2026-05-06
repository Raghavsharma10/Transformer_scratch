def delete(self, upload_id):
        """
        Deletes an upload by ID.
        """

        return super(UploadsProxy, self).delete(upload_id, file_upload=True)