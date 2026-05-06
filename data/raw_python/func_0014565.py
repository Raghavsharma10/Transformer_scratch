def _end_session(self, kill=None):
        """End the client session."""
        if self._stc:
            if kill is None:
                kill = os.environ.get('STC_SESSION_TERMINATE_ON_DISCONNECT')
                kill = _is_true(kill)
            self._stc.end_session(kill)
            self._stc = None