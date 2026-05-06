def update_file_metadata(self, uri, filename=None, description=None,
                             mtime=None, privacy=None):
        """Update file metadata.

        uri -- MediaFire file URI

        Supplying the following keyword arguments would change the
        metadata on the server side:

        filename -- rename file
        description -- set file description string
        mtime -- set file modification time
        privacy -- set file privacy - 'private' or 'public'
        """

        resource = self.get_resource_by_uri(uri)

        if not isinstance(resource, File):
            raise ValueError('Expected File, got {}'.format(type(resource)))

        result = self.api.file_update(resource['quickkey'], filename=filename,
                                      description=description,
                                      mtime=mtime, privacy=privacy)

        return result