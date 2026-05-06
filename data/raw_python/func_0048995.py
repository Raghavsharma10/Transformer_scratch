def get_activity_lookup_session_for_objective_bank(self, objective_bank_id, proxy):
        """Gets the ``OsidSession`` associated with the activity lookup service for the given objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of the
                objective bank
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ActivityLookupSession) - an
                ``ActivityLookupSession``
        raise:  NotFound - ``objective_bank_id`` not found
        raise:  NullArgument - ``objective_bank_id`` or ``proxy`` is
                ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_activity_lookup()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_activity_lookup()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_activity_lookup():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ActivityLookupSession(objective_bank_id, proxy, self._runtime)