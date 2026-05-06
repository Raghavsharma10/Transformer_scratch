def get_validate_upload_form_kwargs(self):
        """
        Return the keyword arguments for instantiating the form for validating
        the upload.

        """

        kwargs = {
            'storage': self.get_storage(),
            'upload_to': self.get_upload_to(),
            'content_type_prefix': self.get_content_type_prefix(),
            'process_to': self.get_process_to(),
            'processed_key_generator': self.get_processed_key_generator(),
        }

        # ``data`` may be provided by a POST from the JavaScript if using a
        # DropZone form, or as querystrings on a redirect GET request from
        # Amazon if not.
        data = {
            'bucket_name': self._get_bucket_name(),
            'key_name': self._get_key_name(),
            'etag': self._get_etag(),
        }
        kwargs.update({'data': data})
        return kwargs