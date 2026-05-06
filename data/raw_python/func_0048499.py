def get_trust_id(self):
        """Gets the ``Trust``  ``Id`` for this authorization.

        return: (osid.id.Id) - the trust ``Id``
        raise:  IllegalState - ``has_trust()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['trustId']):
            raise errors.IllegalState('this Authorization has no trust')
        else:
            return Id(self._my_map['trustId'])