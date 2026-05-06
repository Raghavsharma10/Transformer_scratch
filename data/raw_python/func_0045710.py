def get_bins(self):
        """Gets the bin list resulting from the search.

        return: (osid.resource.BinList) - the bin list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.BinList(self._results, runtime=self._runtime)