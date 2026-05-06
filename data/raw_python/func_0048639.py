def get_type_admin_session(self):
        """Gets the OsidSession associated with the type admin service.

        return: (osid.type.TypeAdminSession) - a TypeAdminSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_type_admin() is false
        compliance: optional - This method must be implemented if
                    supports_type_admin() is true.

        """
        pass
        if not self.supports_type_admin():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise  # OperationFailed()
        try:
            session = sessions.TypeAdminSession()
        except AttributeError:
            raise  # OperationFailed()
        return session