def get_grade_systems(self):
        """Gets the grade system list resulting from the search.

        return: (osid.grading.GradeSystemList) - the grade system list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.GradeSystemList(self._results, runtime=self._runtime)