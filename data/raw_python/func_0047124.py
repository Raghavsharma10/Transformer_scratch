def _set_book_view(self, session):
        """Sets the underlying book view to match current view"""
        if self._book_view == COMPARATIVE:
            try:
                session.use_comparative_book_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_plenary_book_view()
            except AttributeError:
                pass