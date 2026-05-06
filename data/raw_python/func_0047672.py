def get_log_entry_admin_session_for_log(self, log_id):
        """Gets the ``OsidSession`` associated with the log entry administrative service for the given log.

        arg:    log_id (osid.id.Id): the ``Id`` of the ``Log``
        return: (osid.logging.LogEntryAdminSession) - a
                ``LogEntryAdminSession``
        raise:  NotFound - no ``Log`` found by the given ``Id``
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_log_entry_admin()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_log_entry_admin()`` and
        ``supports_visible_federation()`` are ``true``*

        """
        if not self.supports_log_entry_admin():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.LogEntryAdminSession(log_id, runtime=self._runtime)