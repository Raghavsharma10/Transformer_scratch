def extract(self, disk, files, path='.'):
        """Extracts the given files from the given disk.

        Disk must be an integer (1 or 2) indicating from which of the two disks
        to extract.

        Files must be a list of dictionaries containing
        the keys 'path' and 'sha1'.

        Files will be extracted in path and will be named with their sha1.

        Returns a dictionary.

            {'extracted_files': [<sha1>, <sha1>],
             'extraction_errors': [<sha1>, <sha1>]}

        """
        self.logger.debug("Extracting files.")
        extracted_files, failed = self._extract_files(disk, files, path)

        return {'extracted_files': [f for f in extracted_files.keys()],
                'extraction_errors': [f for f in failed.keys()]}