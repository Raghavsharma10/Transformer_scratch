def refresh(self):
        """Refresh the dev_info data used by get_value.

        Only needed if you're not using subscriptions.
        """
        j = self.vera_request(id='sdata', output_format='json').json()
        devices = j.get('devices')
        for device_data in devices:
            if device_data.get('id') == self.device_id:
                self.update(device_data)