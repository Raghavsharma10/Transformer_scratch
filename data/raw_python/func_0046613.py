def get_gradebook_hierarchy_design_session(self, proxy):
        """Gets the session designing gradebook hierarchies.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradebookHierarchyDesignSession) - a
                ``GradebookHierarchyDesignSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_gradebook_hierarchy_design()
                is false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_hierarchy_design()`` is true.*

        """
        if not self.supports_gradebook_hierarchy_design():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradebookHierarchyDesignSession(proxy=proxy, runtime=self._runtime)