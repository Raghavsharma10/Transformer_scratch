def walk(self, root="~/"):
        """Emulation of os.walk behavior against Device Cloud filedata store

        This method will yield tuples in the form ``(dirpath, FileDataDirectory's, FileData's)``
        recursively in pre-order (depth first from top down).

        :param str root: The root path from which the search should commence.  By default, this
            is the root directory for this device cloud account (~).
        :return: Generator yielding 3-tuples of dirpath, directories, and files
        :rtype: 3-tuple in form (dirpath, list of :class:`FileDataDirectory`, list of :class:`FileDataFile`)

        """
        root = validate_type(root, *six.string_types)

        directories = []
        files = []

        # fd_path is real picky
        query_fd_path = root
        if not query_fd_path.endswith("/"):
            query_fd_path += "/"

        for fd_object in self.get_filedata(fd_path == query_fd_path):
            if fd_object.get_type() == "directory":
                directories.append(fd_object)
            else:
                files.append(fd_object)

        # Yield the walk results for this level of the tree
        yield (root, directories, files)

        # recurse on each directory and yield results up the chain
        for directory in directories:
            for dirpath, directories, files in self.walk(directory.get_full_path()):
                yield (dirpath, directories, files)