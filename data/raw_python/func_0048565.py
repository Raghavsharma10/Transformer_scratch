def get_proficiency_lookup_session(self):
        """Gets the OsidSession associated with the proficiency lookup
        service.

        return: (osid.learning.ProficiencyLookupSession) - a
                ProficiencyLookupSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_proficiency_lookup() is false
        compliance: optional - This method must be implemented if
                    supports_proficiency_lookup() is true.

        """
        if not self.supports_proficiency_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ProficiencyLookupSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session