def get_log_entries(self):
        """Gets the log entry list resulting from a search.

        return: (osid.logging.LogEntryList) - the log entry list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.LogEntryList(self._results, runtime=self._runtime)