def get_assessment_id(self):
        """Gets the assessment ``Id`` associated with this learning objective.

        return: (osid.id.Id) - the assessment ``Id``
        raise:  IllegalState - ``has_assessment()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['assessmentId']):
            raise errors.IllegalState('this Objective has no assessment')
        else:
            return Id(self._my_map['assessmentId'])