def _set_family_view(self, session):
        """Sets the underlying family view to match current view"""
        if self._family_view == FEDERATED:
            try:
                session.use_federated_family_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_isolated_family_view()
            except AttributeError:
                pass