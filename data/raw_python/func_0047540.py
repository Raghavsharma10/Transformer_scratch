def _set_gradebook_view(self, session):
        """Sets the underlying gradebook view to match current view"""
        if self._gradebook_view == COMPARATIVE:
            try:
                session.use_comparative_gradebook_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_plenary_gradebook_view()
            except AttributeError:
                pass