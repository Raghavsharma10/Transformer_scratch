def add_auth_hook(self, event):
        """Register event hook on reception of add_auth_hook-event"""

        self.log('Adding authentication hook for', event.authenticator_name)
        self.auth_hooks[event.authenticator_name] = event.event