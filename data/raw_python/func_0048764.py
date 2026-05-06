def match_taking_agent_id(self, agent_id, match):
        """Sets the agent ``Id`` for this query.

        arg:    agent_id (osid.id.Id): an agent ``Id``
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  NullArgument - ``agent_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._add_match('takingAgentId', str(agent_id), bool(match))