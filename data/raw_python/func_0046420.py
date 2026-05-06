def get_bank(self):
        """Gets the ``Bank`` at this node.

        return: (osid.assessment.Bank) - the bank represented by this
                node
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._lookup_session is None:
            mgr = get_provider_manager('ASSESSMENT', runtime=self._runtime, proxy=self._proxy)
            self._lookup_session = mgr.get_bank_lookup_session(proxy=getattr(self, "_proxy", None))
        return self._lookup_session.get_bank(Id(self._my_map['id']))