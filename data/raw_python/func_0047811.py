def use_plenary_objective_bank_view(self):
        """Pass through to provider ObjectiveObjectiveBankSession.use_plenary_objective_bank_view"""
        self._objective_bank_view = PLENARY
        # self._get_provider_session('objective_objective_bank_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_objective_bank_view()
            except AttributeError:
                pass