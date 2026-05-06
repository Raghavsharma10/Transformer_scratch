def _prepare_upload_info(self, source, dest_uri):
        """Prepare Upload object, resolve paths"""

        try:
            dest_resource = self.get_resource_by_uri(dest_uri)
        except ResourceNotFoundError:
            dest_resource = None

        is_fh = hasattr(source, 'read')

        folder_key = None
        name = None

        if dest_resource:
            if isinstance(dest_resource, File):
                folder_key = dest_resource['parent_folderkey']
                name = dest_resource['filename']
            elif isinstance(dest_resource, Folder):
                if is_fh:
                    raise ValueError("Cannot determine target file name")
                basename = posixpath.basename(source)
                dest_uri = posixpath.join(dest_uri, basename)
                try:
                    result = self.get_resource_by_uri(dest_uri)
                    if isinstance(result, Folder):
                        raise ValueError("Target is a folder (file expected)")
                    folder_key = result.get('parent_folderkey', None)
                    name = result['filename']
                except ResourceNotFoundError:
                    # ok, neither a file nor folder, proceed
                    folder_key = dest_resource['folderkey']
                    name = basename
            else:
                raise Exception("Unknown resource type")
        else:
            # get parent resource
            parent_uri = '/'.join(dest_uri.split('/')[0:-1])
            result = self.get_resource_by_uri(parent_uri)
            if not isinstance(result, Folder):
                raise NotAFolderError("Parent component is not a folder")

            folder_key = result['folderkey']
            name = posixpath.basename(dest_uri)

        return folder_key, name