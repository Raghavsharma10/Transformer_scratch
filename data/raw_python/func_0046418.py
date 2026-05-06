def get_assessment_taken_id(self):
        """Gets the ``Id`` of the ``AssessmentTaken``.

        return: (osid.id.Id) - the assessment taken ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective_id
        if not bool(self._my_map['assessmentTakenId']):
            raise errors.IllegalState('assessment_taken empty')
        return Id(self._my_map['assessmentTakenId'])