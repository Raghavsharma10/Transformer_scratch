def has_file_id(self, repository_fname):
        """
        Check for the presence of the given file_id

        :param string repository_fname:
            The file ID
        :return:
            True if we have a :class:`meteorpi_model.FileRecord` with this ID, False otherwise
        """
        self.con.execute('SELECT 1 FROM archive_files WHERE repositoryFname = %s', (repository_fname,))
        return len(self.con.fetchall()) > 0