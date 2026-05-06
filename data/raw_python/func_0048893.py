def get_hierarchies(self):
        """Gets the hierarchy list resulting from the search.

        return: (osid.hierarchy.HierarchyList) - the hierarchy list
        raise:  IllegalState - the hierarchy list was already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.HierarchyList(self._results, runtime=self._runtime)