def _set_hierarchy_view(self, session):
        """Sets the underlying hierarchy view to match current view"""
        if self._hierarchy_view == COMPARATIVE:
            try:
                session.use_comparative_hierarchy_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_plenary_hierarchy_view()
            except AttributeError:
                pass