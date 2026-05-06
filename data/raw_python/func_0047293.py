def get_assessment_part_id(self):
        """Gets the assessment part ``Id`` to which this rule belongs.

        return: (osid.id.Id) - ``Id`` of an assessment part
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective_id
        if not bool(self._my_map['assessmentPartId']):
            raise errors.IllegalState('assessment_part empty')
        return Id(self._my_map['assessmentPartId'])