def get_file(self, filename):
        """Get a file from the repo.

        Returns a file-like stream with the data.
        """
        log.debug('[%s]: reading: //%s/%s', self.name, self.name, filename)
        try:
            blob = self.repo.head.commit.tree/filename
            return blob.data_stream
        except KeyError as err:
            raise GitError(err)