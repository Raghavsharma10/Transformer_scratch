def get_assessments_offered(self):
        """Gets the assessment offered list resulting from the search.

        return: (osid.assessment.AssessmentOfferedList) - the assessment
                offered list
        raise:  IllegalState - the assessment offered list has already
                been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.AssessmentOfferedList(self._results, runtime=self._runtime)