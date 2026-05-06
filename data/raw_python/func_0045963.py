def get_repository(self):
        """Gets the ``Repository`` at this node.

        return: (osid.repository.Repository) - the repository
                represented by this node
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._lookup_session is None:
            mgr = get_provider_manager('REPOSITORY', runtime=self._runtime, proxy=self._proxy)
            self._lookup_session = mgr.get_repository_lookup_session(proxy=getattr(self, "_proxy", None))
        return self._lookup_session.get_repository(Id(self._my_map['id']))