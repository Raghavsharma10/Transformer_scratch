def delete_resource(self, uri, purge=False):
        """Delete file or folder

        uri -- mediafire URI

        Keyword arguments:
        purge -- delete the resource without sending it to Trash.
        """
        try:
            resource = self.get_resource_by_uri(uri)
        except ResourceNotFoundError:
            # Nothing to remove
            return None

        if isinstance(resource, File):
            result = self.delete_file(uri, purge)
        elif isinstance(resource, Folder):
            result = self.delete_folder(uri, purge)
        else:
            raise ValueError('Unsupported resource: {}'.format(type(resource)))

        return result