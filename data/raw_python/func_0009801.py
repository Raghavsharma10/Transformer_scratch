def set_switch_state(self, state):
        """Set the switch state, also update local state."""
        self.set_service_value(
            self.switch_service,
            'Target',
            'newTargetValue',
            state)
        self.set_cache_value('Status', state)