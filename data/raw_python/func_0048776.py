def match_agent_id(self, agent_id, match):
        """Matches the agent identified by the given ``Id``.

        arg:    agent_id (osid.id.Id): the Id of the ``Agent``
        arg:    match (boolean): ``true`` if a positive match, ``false``
                for a negative match
        raise:  NullArgument - ``agent_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._add_match('agentId', str(agent_id), bool(match))