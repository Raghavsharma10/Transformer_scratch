def _set_hierarchy_view(self, session):
        """Sets the underlying hierarchy view to match current view"""
        if self._hierarchy_view == FEDERATED:
            try:
                session.use_federated_hierarchy_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_isolated_hierarchy_view()
            except AttributeError:
                pass