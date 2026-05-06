def extract(self, destination):
        """Extracts the contents of the archive to the specifed directory.

        Args:
            destination (str):
                Path to an empty directory to extract the files to.
        """

        if os.path.exists(destination):
            raise OSError(20, 'Destination exists', destination)

        self.__extract_directory(
            '.',
            self.files['files'],
            destination
        )