def get_resource_agent_session_for_bin(self, bin_id, proxy):
        """Gets a resource agent session for the given bin.

        arg:    bin_id (osid.id.Id): the ``Id`` of the bin
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceAgentSession) - a
                ``ResourceAgentSession``
        raise:  NotFound - ``bin_id`` not found
        raise:  NullArgument - ``bin_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_agent()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_agent()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_resource_agent():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ResourceAgentSession(bin_id, proxy, self._runtime)