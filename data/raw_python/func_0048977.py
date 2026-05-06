def get_proficiency_query_session(self):
        """Gets the ``OsidSession`` associated with the proficiency query service.

        return: (osid.learning.ProficiencyQuerySession) - a
                ``ProficiencyQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_proficiency_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_proficiency_query()`` is ``true``.*

        """
        if not self.supports_proficiency_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ProficiencyQuerySession(runtime=self._runtime)