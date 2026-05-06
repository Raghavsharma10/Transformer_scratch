def delete_file(self, uri, purge=False):
        """Delete file.

        uri -- MediaFire file URI

        Keyword arguments:
        purge -- delete the file without sending it to Trash.
        """
        try:
            resource = self.get_resource_by_uri(uri)
        except ResourceNotFoundError:
            # Nothing to remove
            return None

        if not isinstance(resource, File):
            raise ValueError("File expected, got {}".format(type(resource)))

        if purge:
            func = self.api.file_purge
        else:
            func = self.api.file_delete

        return func(resource['quickkey'])