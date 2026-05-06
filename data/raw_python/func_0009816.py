def get_current_temperature(self, refresh=False):
        """Get current temperature"""
        if refresh:
            self.refresh()
        try:
            return float(self.get_value('temperature'))
        except (TypeError, ValueError):
            return None