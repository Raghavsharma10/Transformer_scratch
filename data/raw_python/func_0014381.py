def create(self, file_or_path, **kwargs):
        """
        Creates an upload for the given file or path.
        """

        opened = False
        if isinstance(file_or_path, str_type()):
            file_or_path = open(file_or_path, 'rb')
            opened = True
        elif not getattr(file_or_path, 'read', False):
            raise Exception("A file or path to a file is required for this operation.")

        try:
            return self.client._post(
                self._url(),
                file_or_path,
                headers=self._resource_class.create_headers({}),
                file_upload=True
            )
        finally:
            if opened:
                file_or_path.close()