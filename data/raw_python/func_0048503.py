def get_qualifier_id(self):
        """Gets the ``Qualifier Id`` for this authorization.

        return: (osid.id.Id) - the qualifier ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective_id
        if not bool(self._my_map['qualifierId']):
            raise errors.IllegalState('qualifier empty')
        return Id(self._my_map['qualifierId'])