def _set_repository_view(self, session):
        """Sets the underlying repository view to match current view"""
        if self._repository_view == COMPARATIVE:
            try:
                session.use_comparative_repository_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_plenary_repository_view()
            except AttributeError:
                pass