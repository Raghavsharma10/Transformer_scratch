def get_gradebook_hierarchy_session(self, proxy):
        """Gets the session traversing gradebook hierarchies.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradebookHierarchySession) - a
                ``GradebookHierarchySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_gradebook_hierarchy() is
                false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_hierarchy()`` is true.*

        """
        if not self.supports_gradebook_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradebookHierarchySession(proxy=proxy, runtime=self._runtime)