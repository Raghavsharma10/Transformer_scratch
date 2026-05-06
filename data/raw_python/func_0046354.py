def get_level_id(self):
        """Gets the ``Id`` of a ``Grade`` corresponding to the assessment difficulty.

        return: (osid.id.Id) - a grade ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['levelId']):
            raise errors.IllegalState('this Assessment has no level')
        else:
            return Id(self._my_map['levelId'])