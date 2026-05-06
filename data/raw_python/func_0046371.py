def get_score_system_id(self):
        """Gets the grade system ``Id`` for the score.

        return: (osid.id.Id) - the grade system ``Id``
        raise:  IllegalState - ``is_scored()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['scoreSystemId']):
            raise errors.IllegalState('this AssessmentOffered has no score_system')
        else:
            return Id(self._my_map['scoreSystemId'])