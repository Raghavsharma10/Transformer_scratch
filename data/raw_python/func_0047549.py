def _set_gradebook_view(self, session):
        """Sets the underlying gradebook view to match current view"""
        if self._gradebook_view == FEDERATED:
            try:
                session.use_federated_gradebook_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_isolated_gradebook_view()
            except AttributeError:
                pass