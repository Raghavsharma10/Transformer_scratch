def get_services_with_resources(cls, request=None):
        """ Get a list of services and resources endpoints.
            {
                ...
                "GitLab": {
                    "url": "/api/gitlab/",
                    "service_project_link_url": "/api/gitlab-service-project-link/",
                    "resources": {
                        "Project": "/api/gitlab-projects/",
                        "Group": "/api/gitlab-groups/"
                    }
                },
                ...
            }
        """
        from django.apps import apps

        data = {}
        for service in cls._registry.values():
            service_model = apps.get_model(service['model_name'])
            service_project_link = service_model.projects.through
            service_project_link_url = reverse(cls.get_list_view_for_model(service_project_link), request=request)

            data[service['name']] = {
                'url': reverse(service['list_view'], request=request),
                'service_project_link_url': service_project_link_url,
                'resources': {resource['name']: reverse(resource['list_view'], request=request)
                              for resource in service['resources'].values()},
                'properties': {resource['name']: reverse(resource['list_view'], request=request)
                               for resource in service.get('properties', {}).values()},
                'is_public_service': cls.is_public_service(service_model)
            }
        return data