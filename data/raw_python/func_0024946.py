def _find_service_name(self):
        """
        For cloud operations there is support for multiple pools of resources
        dedicated to logstash.  The service name as a result follows the
        pattern logstash-{n} where n is some number.  We can find it from the
        service marketplace.
        """
        space = predix.admin.cf.spaces.Space()
        services = space.get_services()
        for service in services:
            if service.startswith('logstash'):
                return service

        return 'logstash-3'