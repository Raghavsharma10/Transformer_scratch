def get_gradebooks(self):
        """Gets the gradebook list resulting from the search.

        return: (osid.grading.GradebookList) - the gradebook list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.GradebookList(self._results, runtime=self._runtime)