def get_objective_admin_session_for_objective_bank(self, objective_bank_id):
        """Gets the ``OsidSession`` associated with the objective admin service for the given objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of the
                objective bank
        return: (osid.learning.ObjectiveAdminSession) - ``an
                _objective_admin_session``
        raise:  NotFound - ``objective_bank_id`` not found
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_objective_admin()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_admin()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_objective_admin():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ObjectiveAdminSession(objective_bank_id, runtime=self._runtime)