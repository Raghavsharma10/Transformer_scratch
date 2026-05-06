def get_resources(self):
        """Gets the resource list resulting from a search.

        return: (osid.resource.ResourceList) - the resource list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.ResourceList(self._results, runtime=self._runtime)