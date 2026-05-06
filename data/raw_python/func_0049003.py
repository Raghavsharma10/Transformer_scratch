def get_proficiency_admin_session_for_objective_bank(self, objective_bank_id, proxy):
        """Gets the ``OsidSession`` associated with the proficiency administration service for the given objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of the
                ``ObjectiveBank``
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ProficiencyAdminSession) - a
                ``ProficiencyAdminSession``
        raise:  NotFound - no objective bank found by the given ``Id``
        raise:  NullArgument - ``objective_bank_id`` or ``proxy`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_proficiency_admin()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_proficiency_admin()`` and
        ``supports_visible_federation()`` are ``true``*

        """
        if not self.supports_proficiency_admin():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ProficiencyAdminSession(objective_bank_id, proxy, self._runtime)