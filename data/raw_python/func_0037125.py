def get_upload_key(self):
        """Get the `Key` from the S3 bucket for the uploaded file.

        :returns: Key (object) of the uploaded file.
        :rtype: :py:class:`boto.s3.key.Key`

        """

        if not hasattr(self, '_upload_key'):
            self._upload_key = self.get_storage().bucket.get_key(
                self.cleaned_data['key_name'])
        return self._upload_key