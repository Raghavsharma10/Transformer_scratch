def get_repositories(self):
        """Gets the repository list resulting from the search.

        return: (osid.repository.RepositoryList) - the repository list
        raise:  IllegalState - the list has already been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.RepositoryList(self._results, runtime=self._runtime)