def get_bank_admin_session(self):
        """Gets the OsidSession associated with the bank administration service.

        return: (osid.assessment.BankAdminSession) - a
                ``BankAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bank_admin() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_bank_admin()`` is true.*

        """
        if not self.supports_bank_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BankAdminSession(runtime=self._runtime)