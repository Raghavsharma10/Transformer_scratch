def data_complete(self):
        """
        Return True if all the expected datadir files are present
        """
        return task.data_complete(self.datadir, self.sitedir,
            self._get_container_name)