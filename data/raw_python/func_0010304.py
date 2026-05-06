def delete_file(self, path):
        """Delete a file or directory from the filedata store

        This method removes a file or directory (recursively) from
        the filedata store.

        :param path: The path of the file or directory to remove
            from the file data store.

        """
        path = validate_type(path, *six.string_types)
        if not path.startswith("/"):
            path = "/" + path

        self._conn.delete("/ws/FileData{path}".format(path=path))