def _set_repository_view(self, session):
        """Sets the underlying repository view to match current view"""
        if self._repository_view == FEDERATED:
            try:
                session.use_federated_repository_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_isolated_repository_view()
            except AttributeError:
                pass