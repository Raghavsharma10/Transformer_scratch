def use_plenary_bank_view(self):
        """Pass through to provider ItemBankSession.use_plenary_bank_view"""
        self._bank_view = PLENARY
        # self._get_provider_session('item_bank_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_bank_view()
            except AttributeError:
                pass