def vera_request(self, **kwargs):
        """Perfom a vera_request for this device."""
        request_payload = {
            'output_format': 'json',
            'DeviceNum': self.device_id,
        }
        request_payload.update(kwargs)

        return self.vera_controller.data_request(request_payload)