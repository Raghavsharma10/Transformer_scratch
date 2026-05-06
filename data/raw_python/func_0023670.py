def get_resource_by_key(self, resource_key):
        """Return resource by quick_key/folder_key.

        key -- quick_key or folder_key
        """

        # search for quick_key by default
        lookup_order = ["quick_key", "folder_key"]

        if len(resource_key) == FOLDER_KEY_LENGTH:
            lookup_order = ["folder_key", "quick_key"]

        resource = None

        for lookup_key in lookup_order:
            try:
                if lookup_key == "folder_key":
                    info = self.api.folder_get_info(folder_key=resource_key)
                    resource = Folder(info['folder_info'])
                elif lookup_key == "quick_key":
                    info = self.api.file_get_info(quick_key=resource_key)
                    resource = File(info['file_info'])
            except MediaFireApiError:
                # TODO: Check response code
                pass

            if resource:
                break

        if not resource:
            raise ResourceNotFoundError(resource_key)

        return resource