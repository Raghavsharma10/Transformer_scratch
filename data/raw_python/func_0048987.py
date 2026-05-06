def get_objective_hierarchy_design_session_for_objective_bank(self, objective_bank_id, proxy):
        """Gets the ``OsidSession`` associated with the objective hierarchy design service for the given objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of the
                objective bank
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveHierarchyDesignSession) - an
                ``ObjectiveHierarchyDesignSession``
        raise:  NotFound - ``objective_bank_id`` not found
        raise:  NullArgument - ``objective_bank_id`` or ``proxy`` is
                ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented -
                ``supports_objective_hierarchy_design()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_hierarchy_design()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_objective_hierarchy_design():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ObjectiveHierarchyDesignSession(objective_bank_id, proxy, self._runtime)