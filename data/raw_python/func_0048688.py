def get_bin(self):
        """Gets the ``Bin`` at this node.

        return: (osid.resource.Bin) - the bin represented by this node
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._lookup_session is None:
            mgr = get_provider_manager('RESOURCE', runtime=self._runtime, proxy=self._proxy)
            self._lookup_session = mgr.get_bin_lookup_session(proxy=getattr(self, "_proxy", None))
        return self._lookup_session.get_bin(Id(self._my_map['id']))