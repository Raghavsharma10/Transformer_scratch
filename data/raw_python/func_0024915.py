def get_service_plan_guid(self, service_name, plan_name):
        """
        Return the service plan GUID for the given service / plan.
        """
        for plan in self.get_service_plan_for_service(service_name):
            if plan['entity']['name'] == plan_name:
                return plan['metadata']['guid']

        return None