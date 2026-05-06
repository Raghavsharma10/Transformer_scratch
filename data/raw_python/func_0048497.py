def get_resource_id(self):
        """Gets the ``resource _id`` for this authorization.

        return: (osid.id.Id) - the ``Resource Id``
        raise:  IllegalState - ``has_resource()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['resourceId']):
            raise errors.IllegalState('this Authorization has no resource')
        else:
            return Id(self._my_map['resourceId'])