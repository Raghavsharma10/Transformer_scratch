def get_bin_hierarchy_design_session(self):
        """Gets the bin hierarchy design session.

        return: (osid.resource.BinHierarchyDesignSession) - a
                ``BinHierarchyDesignSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bin_hierarchy_design()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_bin_hierarchy_design()`` is ``true``.*

        """
        if not self.supports_bin_hierarchy_design():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BinHierarchyDesignSession(runtime=self._runtime)