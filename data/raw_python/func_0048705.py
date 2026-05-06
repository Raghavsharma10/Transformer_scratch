def get_agent_id(self):
        """Gets the agent ``Id`` who created this entry.

        return: (osid.id.Id) - the agent ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['agentId']):
            raise errors.IllegalState('this LogEntry has no agent')
        else:
            return Id(self._my_map['agentId'])