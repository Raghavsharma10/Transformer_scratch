def get_assessments_taken(self):
        """Gets the assessment taken list resulting from the search.

        return: (osid.assessment.AssessmentTakenList) - the assessment
                taken list
        raise:  IllegalState - the assessment taken list has already
                been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.AssessmentTakenList(self._results, runtime=self._runtime)