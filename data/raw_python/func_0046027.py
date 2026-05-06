def get_grade_entries(self):
        """Gets the package list resulting from the search.

        return: (osid.grading.GradeEntryList) - the grade entry list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.GradeEntryList(self._results, runtime=self._runtime)