def get_agent_lookup_session_for_agency(self, agency_id):
        """Gets the ``OsidSession`` associated with the agent lookup service for the given agency.

        arg:    agency_id (osid.id.Id): the ``Id`` of the agency
        return: (osid.authentication.AgentLookupSession) - ``an
                _agent_lookup_session``
        raise:  NotFound - ``agency_id`` not found
        raise:  NullArgument - ``agency_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_agent_lookup()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_agent_lookup()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_agent_lookup():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AgentLookupSession(agency_id, runtime=self._runtime)