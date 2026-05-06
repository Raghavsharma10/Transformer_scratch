def get_assessment_part(self):
        """Gets the parent assessment.

        return: (osid.assessment.authoring.AssessmentPart) - the parent
                assessment part
        raise:  IllegalState - ``has_parent_part()`` is ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        if not self.has_parent_part():
            raise errors.IllegalState('no parent part')
        lookup_session = self._get_assessment_part_lookup_session()
        return lookup_session.get_assessment_part(self.get_assessment_part_id())