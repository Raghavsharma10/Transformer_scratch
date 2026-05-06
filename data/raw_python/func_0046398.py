def get_taker_id(self):
        """Gets the ``Id`` of the resource who took or is taking this assessment.

        return: (osid.id.Id) - the resource ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._my_map['takerId']:
            return Id(self._my_map['takerId'])
        else:
            return Id(self._my_map['takingAgentId'])