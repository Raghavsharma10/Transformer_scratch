def get_bin_hierarchy_session(self):
        """Gets the bin hierarchy traversal session.

        return: (osid.resource.BinHierarchySession) - ``a
                BinHierarchySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bin_hierarchy()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_bin_hierarchy()`` is ``true``.*

        """
        if not self.supports_bin_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BinHierarchySession(runtime=self._runtime)