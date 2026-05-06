def get_compositions(self):
        """Gets the composition list resulting from a search.

        return: (osid.repository.CompositionList) - the composition list
        raise:  IllegalState - the list has already been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.CompositionList(self._results, runtime=self._runtime)