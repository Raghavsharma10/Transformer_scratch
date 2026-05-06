def get_strict_value(self, name):
        """Get a case-sensitive keys value from the dev_info area.
        """
        dev_info = self.json_state.get('deviceInfo')
        return dev_info.get(name, None)