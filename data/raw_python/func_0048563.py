def get_activity_admin_session(self):
        """Gets the OsidSession associated with the activity administration
        service.

        return: (osid.learning.ActivityAdminSession) - a
                ActivityAdminSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_activity_admin() is false
        compliance: optional - This method must be implemented if
                    supports_activity_admin() is true.

        """
        if not self.supports_activity_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ActivityAdminSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session