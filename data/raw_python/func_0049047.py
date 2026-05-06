def get_book(self):
        """Gets the ``Book`` at this node.

        return: (osid.commenting.Book) - the book represented by this
                node
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._lookup_session is None:
            mgr = get_provider_manager('COMMENTING', runtime=self._runtime, proxy=self._proxy)
            self._lookup_session = mgr.get_book_lookup_session(proxy=getattr(self, "_proxy", None))
        return self._lookup_session.get_book(Id(self._my_map['id']))