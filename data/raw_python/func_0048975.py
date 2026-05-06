def get_activity_query_session(self):
        """Gets the ``OsidSession`` associated with the activity query service.

        return: (osid.learning.ActivityQuerySession) - a
                ``ActivityQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_activity_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_activity_query()`` is ``true``.*

        """
        if not self.supports_activity_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ActivityQuerySession(runtime=self._runtime)