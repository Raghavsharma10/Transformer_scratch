def get_function_id(self):
        """Gets the ``Function Id`` for this authorization.

        return: (osid.id.Id) - the function ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective_id
        if not bool(self._my_map['functionId']):
            raise errors.IllegalState('function empty')
        return Id(self._my_map['functionId'])