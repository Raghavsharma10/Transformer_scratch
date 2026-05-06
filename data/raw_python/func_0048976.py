def get_activity_query_session_for_objective_bank(self, objective_bank_id):
        """Gets the ``OsidSession`` associated with the activity query service for the given objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of the
                objective bank
        return: (osid.learning.ActivityQuerySession) - an
                ``ActivityQuerySession``
        raise:  NotFound - ``objective_bank_id`` not found
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_activity_query()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_activity_query()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_activity_query():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ActivityQuerySession(objective_bank_id, runtime=self._runtime)