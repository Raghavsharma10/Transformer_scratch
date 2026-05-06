def update(self, params):
        """Update the dev_info data from a dictionary.

        Only updates if it already exists in the device.
        """
        dev_info = self.json_state.get('deviceInfo')
        dev_info.update({k: params[k] for k in params if dev_info.get(k)})