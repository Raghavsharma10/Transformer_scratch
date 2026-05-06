def _set_family_view(self, session):
        """Sets the underlying family view to match current view"""
        if self._family_view == COMPARATIVE:
            try:
                session.use_comparative_family_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_plenary_family_view()
            except AttributeError:
                pass