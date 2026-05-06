def _generate_name(self, space, service_name, plan_name):
        """
        Can generate a name based on the space, service name and plan.
        """
        return str.join('-', [space, service_name, plan_name]).lower()