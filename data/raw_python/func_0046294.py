def get_bank_hierarchy_session(self):
        """Gets the session traversing bank hierarchies.

        return: (osid.assessment.BankHierarchySession) - a
                ``BankHierarchySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bank_hierarchy() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_bank_hierarchy()`` is true.*

        """
        if not self.supports_bank_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BankHierarchySession(runtime=self._runtime)