def has_service_of_type(self, service_type):
        """
        Tests whether a service instance exists for the given
        service.
        """
        summary = self.get_space_summary()
        for instance in summary['services']:
            if 'service_plan' in instance:
                if service_type == instance['service_plan']['service']['label']:
                    return True

        return False