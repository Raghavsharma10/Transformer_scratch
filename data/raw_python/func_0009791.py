def set_service_value(self, service_id, set_name, parameter_name, value):
        """Set a variable on the vera device.

        This will call the Vera api to change device state.
        """
        payload = {
            'id': 'lu_action',
            'action': 'Set' + set_name,
            'serviceId': service_id,
            parameter_name: value
        }
        result = self.vera_request(**payload)
        logger.debug("set_service_value: "
                  "result of vera_request with payload %s: %s",
                  payload, result.text)