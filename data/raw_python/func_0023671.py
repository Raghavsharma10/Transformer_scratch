def get_resource_by_path(self, path, folder_key=None):
        """Return resource by remote path.

        path -- remote path

        Keyword arguments:
        folder_key -- what to use as the root folder (None for root)
        """
        logger.debug("resolving %s", path)

        # remove empty path components
        path = posixpath.normpath(path)
        components = [t for t in path.split(posixpath.sep) if t != '']

        if not components:
            # request for root
            return Folder(
                self.api.folder_get_info(folder_key)['folder_info']
            )

        resource = None

        for component in components:
            exists = False
            for item in self._folder_get_content_iter(folder_key):
                name = item['name'] if 'name' in item else item['filename']

                if name == component:
                    exists = True
                    if components[-1] != component:
                        # still have components to go through
                        if 'filename' in item:
                            # found a file, expected a directory
                            raise NotAFolderError(item['filename'])
                        folder_key = item['folderkey']
                    else:
                        # found the leaf
                        resource = item
                    break

                if resource is not None:
                    break

            if not exists:
                # intermediate component does not exist - bailing out
                break

        if resource is None:
            raise ResourceNotFoundError(path)

        if "quickkey" in resource:
            file_info = self.api.file_get_info(
                resource['quickkey'])['file_info']
            result = File(file_info)
        elif "folderkey" in resource:
            folder_info = self.api.folder_get_info(
                resource['folderkey'])['folder_info']
            result = Folder(folder_info)

        return result