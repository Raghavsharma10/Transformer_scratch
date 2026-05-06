def _create_service(self, parameters={}, **kwargs):
        """
        Create a Cloud Foundry service that has custom parameters.
        """
        logging.debug("_create_service()")
        logging.debug(str.join(',', [self.service_name, self.plan_name,
            self.name, str(parameters)]))

        return self.service.create_service(self.service_name, self.plan_name,
                self.name, parameters, **kwargs)