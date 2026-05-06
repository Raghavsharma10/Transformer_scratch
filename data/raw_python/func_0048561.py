def get_activity_search_session(self):
        """Gets the OsidSession associated with the activity search
        service.

        return: (osid.learning.ActivitySearchSession) - a
                ActivitySearchSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_activity_search() is false
        compliance: optional - This method must be implemented if
                    supports_activity_search() is true.

        """
        if not self.supports_activity_search():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ActivitySearchSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session