def get_resource_by_uri(self, uri):
        """Return resource described by MediaFire URI.

        uri -- MediaFire URI

        Examples:
            Folder (using folderkey):
            mf:r5g3p2z0sqs3j
            mf:r5g3p2z0sqs3j/folder/file.ext

            File (using quickkey):
            mf:xkr43dadqa3o2p2

            Path:
            mf:///Documents/file.ext
        """

        location = self._parse_uri(uri)

        if location.startswith("/"):
            # Use path lookup only, root=myfiles
            result = self.get_resource_by_path(location)
        elif "/" in location:
            # mf:abcdefjhijklm/name
            resource_key, path = location.split('/', 2)
            parent_folder = self.get_resource_by_key(resource_key)
            if not isinstance(parent_folder, Folder):
                raise NotAFolderError(resource_key)
            # perform additional lookup by path
            result = self.get_resource_by_path(
                path, folder_key=parent_folder['folderkey'])
        else:
            # mf:abcdefjhijklm
            result = self.get_resource_by_key(location)

        return result