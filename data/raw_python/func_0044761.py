def __extract_file(self, path, fileinfo, destination):
        """Extracts the specified file to the specified destination.

        Args:
            path (str):
                Relative (to the root of the archive) path of the
                file to extract.

            fileinfo (dict):
                Dictionary containing the offset and size of the file
                (Extracted from the header).

            destination (str):
                Directory to extract the archive to.
        """

        if 'offset' not in fileinfo:
            self.__copy_extracted(path, destination)
            return

        self.asarfile.seek(
            self.__absolute_offset(fileinfo['offset'])
        )

        # TODO: read in chunks, ain't going to read multiple GB's in memory
        contents = self.asarfile.read(
            self.__absolute_offset(fileinfo['size'])
        )

        destination_path = os.path.join(destination, path)

        with open(destination_path, 'wb') as fp:
            fp.write(contents)

        LOGGER.debug('Extracted %s to %s', path, destination_path)