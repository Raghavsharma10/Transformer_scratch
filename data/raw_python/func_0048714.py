def get_log(self):
        """Gets the ``Log`` at this node.

        return: (osid.logging.Log) - the log represented by this node
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._lookup_session is None:
            mgr = get_provider_manager('LOGGING', runtime=self._runtime, proxy=self._proxy)
            self._lookup_session = mgr.get_log_lookup_session(proxy=getattr(self, "_proxy", None))
        return self._lookup_session.get_log(Id(self._my_map['id']))