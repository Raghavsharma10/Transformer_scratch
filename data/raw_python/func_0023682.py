def update_folder_metadata(self, uri, foldername=None, description=None,
                               mtime=None, privacy=None,
                               privacy_recursive=None):
        """Update folder metadata.

        uri -- MediaFire file URI

        Supplying the following keyword arguments would change the
        metadata on the server side:

        filename -- rename file
        description -- set file description string
        mtime -- set file modification time
        privacy -- set file privacy - 'private' or 'public'
        recursive -- update folder privacy recursively
        """

        resource = self.get_resource_by_uri(uri)

        if not isinstance(resource, Folder):
            raise ValueError('Expected Folder, got {}'.format(type(resource)))

        result = self.api.folder_update(resource['folderkey'],
                                        foldername=foldername,
                                        description=description,
                                        mtime=mtime,
                                        privacy=privacy,
                                        privacy_recursive=privacy_recursive)

        return result