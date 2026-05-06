def set_agent(self, agent_id):
        """Sets the agent.

        arg:    agent_id (osid.id.Id): the new agent
        raise:  InvalidArgument - ``agent_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``agent_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_agent_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(agent_id):
            raise errors.InvalidArgument()
        self._my_map['agentId'] = str(agent_id)