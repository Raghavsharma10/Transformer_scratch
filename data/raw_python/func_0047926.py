def get_hierarchy_design_session_for_hierarchy(self, hierarchy_id, proxy):
        """Gets the ``OsidSession`` associated with the topology design service using for the given hierarchy.

        arg:    hierarchy_id (osid.id.Id): the ``Id`` of the hierarchy
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.hierarchy.HierarchyDesignSession) - a
                ``HierarchyDesignSession``
        raise:  NotFound - ``hierarchy_id`` is not found
        raise:  NullArgument - ``hierarchy_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_hierarchy_design()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_hierarchy_design()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_hierarchy_design():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.HierarchyDesignSession(hierarchy_id, proxy, self._runtime)