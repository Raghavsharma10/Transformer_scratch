def get_objective_id(self):
        """Gets the ``Id`` of the related objective.

        return: (osid.id.Id) - the objective ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective_id
        if not bool(self._my_map['objectiveId']):
            raise errors.IllegalState('objective empty')
        return Id(self._my_map['objectiveId'])