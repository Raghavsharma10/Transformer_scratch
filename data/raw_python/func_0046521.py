def get_logs(self):
        """Gets the log list resulting from a search.

        return: (osid.logging.LogList) - the log list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.LogList(self._results, runtime=self._runtime)