def _set_bank_view(self, session):
        """Sets the underlying bank view to match current view"""
        if self._bank_view == COMPARATIVE:
            try:
                session.use_comparative_bank_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_plenary_bank_view()
            except AttributeError:
                pass