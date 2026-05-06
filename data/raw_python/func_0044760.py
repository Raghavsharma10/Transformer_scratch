def __extract_directory(self, path, files, destination):
        """Extracts a single directory to the specified directory on disk.

        Args:
            path (str):
                Relative (to the root of the archive) path of the directory
                to extract.

            files (dict):
                A dictionary of files from a *.asar file header.

            destination (str):
                The path to extract the files to.
        """

        # assures the destination directory exists
        destination_path = os.path.join(destination, path)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        for name, contents in files.items():
            item_path = os.path.join(path, name)

            # objects that have a 'files' member are directories,
            # recurse into them
            if 'files' in contents:
                self.__extract_directory(
                    item_path,
                    contents['files'],
                    destination
                )

                continue

            self.__extract_file(item_path, contents, destination)