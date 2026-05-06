def get_service_name_resources(cls, service_name):
        """ Get resource models by service name """
        from django.apps import apps

        resources = cls._registry[service_name]['resources'].keys()
        return [apps.get_model(resource) for resource in resources]