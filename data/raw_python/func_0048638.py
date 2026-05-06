def get_type_lookup_session(self):
        """Gets the OsidSession associated with the type lookup service.

        return: (osid.type.TypeLookupSession) - a TypeLookupSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_type_lookup() is false
        compliance: optional - This method must be implemented if
                    supports_type_lookup() is true.

        """
        if not self.supports_type_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise  # OperationFailed()
        try:
            session = sessions.TypeLookupSession(runtime=self._runtime)
        except AttributeError:
            raise  # OperationFailed()
        return session