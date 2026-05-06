def _set_no_catalog_view(self, session):
        """Sets the underlying no_catalog view to match current view"""
        if self._no_catalog_view == COMPARATIVE:
            try:
                session.use_comparative_no_catalog_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_plenary_no_catalog_view()
            except AttributeError:
                pass