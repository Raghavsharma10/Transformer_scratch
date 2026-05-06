def get_avatar_id(self):
        """Gets the asset ``Id``.

        return: (osid.id.Id) - the asset ``Id``
        raise:  IllegalState - ``has_avatar()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['avatarId']):
            raise errors.IllegalState('this Resource has no avatar')
        else:
            return Id(self._my_map['avatarId'])