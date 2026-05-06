def get_rating_id(self):
        """Gets the ``Id`` of the ``Grade``.

        return: (osid.id.Id) - the ``Agent``  ``Id``
        raise:  IllegalState - ``has_rating()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['ratingId']):
            raise errors.IllegalState('this Comment has no rating')
        else:
            return Id(self._my_map['ratingId'])