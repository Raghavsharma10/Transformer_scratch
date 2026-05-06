def get_rubric_id(self):
        """Gets the ``Id`` of the rubric.

        return: (osid.id.Id) - an assessment taken ``Id``
        raise:  IllegalState - ``has_rubric()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['rubricId']):
            raise errors.IllegalState('this AssessmentTaken has no rubric')
        else:
            return Id(self._my_map['rubricId'])