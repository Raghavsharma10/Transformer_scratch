def _set_book_view(self, session):
        """Sets the underlying book view to match current view"""
        if self._book_view == FEDERATED:
            try:
                session.use_federated_book_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_isolated_book_view()
            except AttributeError:
                pass