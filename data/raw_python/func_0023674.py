def create_folder(self, uri, recursive=False):
        """Create folder.

        uri -- MediaFire URI

        Keyword arguments:
        recursive -- set to True to create intermediate folders.
        """
        logger.info("Creating %s", uri)

        # check that folder exists already
        try:
            resource = self.get_resource_by_uri(uri)

            if isinstance(resource, Folder):
                return resource
            else:
                raise NotAFolderError(uri)
        except ResourceNotFoundError:
            pass

        location = self._parse_uri(uri)

        folder_name = posixpath.basename(location)
        parent_uri = 'mf://' + posixpath.dirname(location)

        try:
            parent_node = self.get_resource_by_uri(parent_uri)
            if not isinstance(parent_node, Folder):
                raise NotAFolderError(parent_uri)
            parent_key = parent_node['folderkey']
        except ResourceNotFoundError:
            if recursive:
                result = self.create_folder(parent_uri, recursive=True)
                parent_key = result['folderkey']
            else:
                raise

        # We specify exact location, so don't allow duplicates
        result = self.api.folder_create(
            folder_name, parent_key=parent_key, action_on_duplicate='skip')

        logger.info("Created folder '%s' [mf:%s]",
                    result['name'], result['folder_key'])

        return self.get_resource_by_key(result['folder_key'])