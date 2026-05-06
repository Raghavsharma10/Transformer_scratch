def get_service_models(cls):
        """ Get a list of service models.
            {
                ...
                'gitlab': {
                    "service": nodeconductor_gitlab.models.GitLabService,
                    "service_project_link": nodeconductor_gitlab.models.GitLabServiceProjectLink,
                    "resources": [
                        nodeconductor_gitlab.models.Group,
                        nodeconductor_gitlab.models.Project
                    ],
                },
                ...
            }

        """
        from django.apps import apps

        data = {}
        for key, service in cls._registry.items():
            service_model = apps.get_model(service['model_name'])
            service_project_link = service_model.projects.through
            data[key] = {
                'service': service_model,
                'service_project_link': service_project_link,
                'resources': [apps.get_model(r) for r in service['resources'].keys()],
                'properties': [apps.get_model(r) for r in service['properties'].keys() if '.' in r],
            }

        return data