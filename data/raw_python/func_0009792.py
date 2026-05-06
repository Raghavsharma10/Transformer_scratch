def call_service(self, service_id, action):
        """Call a Vera service.

        This will call the Vera api to change device state.
        """
        result = self.vera_request(id='action', serviceId=service_id,
                                   action=action)
        logger.debug("call_service: "
                  "result of vera_request with id %s: %s", service_id,
                  result.text)
        return result