def delete_folder(self, uri, purge=False):
        """Delete folder.

        uri -- MediaFire folder URI

        Keyword arguments:
        purge -- delete the folder without sending it to Trash
        """

        try:
            resource = self.get_resource_by_uri(uri)
        except ResourceNotFoundError:
            # Nothing to remove
            return None

        if not isinstance(resource, Folder):
            raise ValueError("Folder expected, got {}".format(type(resource)))

        if purge:
            func = self.api.folder_purge
        else:
            func = self.api.folder_delete

        try:
            result = func(resource['folderkey'])
        except MediaFireApiError as err:
            if err.code == 100:
                logger.debug(
                    "Delete folder returns error 900 but folder is deleted: "
                    "http://forum.mediafiredev.com/showthread.php?129")

                result = {}
            else:
                raise

        return result