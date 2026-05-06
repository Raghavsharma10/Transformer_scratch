def vera_request(self, **kwargs):
        """Perfom a vera_request for this scene."""
        request_payload = {
            'output_format': 'json',
            'SceneNum': self.scene_id,
        }
        request_payload.update(kwargs)

        return self.vera_controller.data_request(request_payload)