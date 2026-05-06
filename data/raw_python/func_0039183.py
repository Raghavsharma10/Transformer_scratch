def compare(self, concurrent=False, identify=False, size=False):
        """Compares the two disks according to flags.

        Generates the following report:

        ::

            {'created_files': [{'path': '/file/in/disk1/not/in/disk0',
                                'sha1': 'sha1_of_the_file'}],
             'deleted_files': [{'path': '/file/in/disk0/not/in/disk1',
                                'original_sha1': 'sha1_of_the_file'}],
             'modified_files': [{'path': '/file/both/disks/but/different',
                                 'sha1': 'sha1_of_the_file_on_disk0',
                                 'original_sha1': 'sha1_of_the_file_on_disk0'}]}

        If concurrent is set to True, the logic will use multiple CPUs to
        speed up the process.

        The identify and size keywords will add respectively the type
        and the size of the files to the results.

        """
        self.logger.debug("Comparing FS contents.")
        results = compare_filesystems(self.filesystems[0], self.filesystems[1],
                                      concurrent=concurrent)

        if identify:
            self.logger.debug("Gatering file types.")
            results = files_type(self.filesystems[0], self.filesystems[1],
                                 results)

        if size:
            self.logger.debug("Gatering file sizes.")
            results = files_size(self.filesystems[0], self.filesystems[1],
                                 results)

        return results