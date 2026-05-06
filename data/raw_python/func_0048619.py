def get_assessments(self):
        """Gets the assessment list resulting from the search.

        return: (osid.assessment.AssessmentList) - the assessment list
        raise:  IllegalState - the assessment list has already been
                retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.AssessmentList(self._results, runtime=self._runtime)