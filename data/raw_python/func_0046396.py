def get_assessment_offered_id(self):
        """Gets the ``Id`` of the ``AssessmentOffered``.

        return: (osid.id.Id) - the assessment offered ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective_id
        if not bool(self._my_map['assessmentOfferedId']):
            raise errors.IllegalState('assessment_offered empty')
        return Id(self._my_map['assessmentOfferedId'])