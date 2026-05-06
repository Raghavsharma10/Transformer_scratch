def _set_objective_bank_view(self, session):
        """Sets the underlying objective_bank view to match current view"""
        if self._objective_bank_view == FEDERATED:
            try:
                session.use_federated_objective_bank_view()
            except AttributeError:
                pass
        else:
            try:
                session.use_isolated_objective_bank_view()
            except AttributeError:
                pass