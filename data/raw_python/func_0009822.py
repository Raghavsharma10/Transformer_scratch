def activate(self):
        """Activate a Vera scene.

        This will call the Vera api to activate a scene.
        """
        payload = {
            'id': 'lu_action',
            'action': 'RunScene',
            'serviceId': self.scene_service
        }
        result = self.vera_request(**payload)
        logger.debug("activate: "
                  "result of vera_request with payload %s: %s",
                  payload, result.text)

        self._active = True