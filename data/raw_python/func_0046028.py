def get_gradebook_columns(self):
        """Gets the gradebook column list resulting from the search.

        return: (osid.grading.GradebookColumnList) - the gradebook
                column list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.GradebookColumnList(self._results, runtime=self._runtime)