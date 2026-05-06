def progress(self):
        """Returned a cached ProcessLogger to record build progress """

        if not self._progress:

            # If won't be building, only use one connection
            new_connection = False if self._library.read_only else True

            self._progress = ProcessLogger(self.dataset, self.logger, new_connection=new_connection)

        return self._progress