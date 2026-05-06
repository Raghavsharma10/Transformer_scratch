def timed_connectivity_check(self, event):
        """Tests internet connectivity in regular intervals and updates the nodestate accordingly"""
        self.status = self._can_connect()
        self.log('Timed connectivity check:', self.status, lvl=verbose)

        if self.status:
            if not self.old_status:
                self.log('Connectivity gained')
                self.fireEvent(backend_nodestate_toggle(STATE_UUID_CONNECTIVITY, on=True, force=True))
        else:
            if self.old_status:
                self.log('Connectivity lost', lvl=warn)
                self.old_status = False
                self.fireEvent(backend_nodestate_toggle(STATE_UUID_CONNECTIVITY, off=True, force=True))

        self.old_status = self.status