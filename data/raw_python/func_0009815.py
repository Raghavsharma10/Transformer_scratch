def get_current_goal_temperature(self, refresh=False):
        """Get current goal temperature / setpoint"""
        if refresh:
            self.refresh()
        try:
            return float(self.get_value('setpoint'))
        except (TypeError, ValueError):
            return None