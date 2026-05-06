def get_folder_contents_iter(self, uri):
        """Return iterator for directory contents.

        uri -- mediafire URI

        Example:

            for item in get_folder_contents_iter('mf:///Documents'):
                print(item)
        """
        resource = self.get_resource_by_uri(uri)

        if not isinstance(resource, Folder):
            raise NotAFolderError(uri)

        folder_key = resource['folderkey']

        for item in self._folder_get_content_iter(folder_key):
            if 'filename' in item:
                # Work around https://mediafire.mantishub.com/view.php?id=5
                # TODO: remove in 1.0
                if ".patch." in item['filename']:
                    continue
                yield File(item)
            elif 'name' in item:
                yield Folder(item)