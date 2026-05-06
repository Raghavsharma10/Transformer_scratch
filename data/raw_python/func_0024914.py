def get_service_plan_for_service(self, service_name):
        """
        Return the service plans available for a given service.
        """
        services = self.get_services()
        for service in services['resources']:
            if service['entity']['label'] == service_name:
                response = self.api.get(service['entity']['service_plans_url'])
                return response['resources']