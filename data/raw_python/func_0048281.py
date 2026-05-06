def get_resource_bin_assignment_session(self, proxy):
        """Gets the session for assigning resource to bin mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceBinAssignmentSession) - a
                ``ResourceBinAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_bin_assignment()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_bin_assignment()`` is ``true``.*

        """
        if not self.supports_resource_bin_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceBinAssignmentSession(proxy=proxy, runtime=self._runtime)