def get_resource_models(cls):
        """ Get a list of resource models.
            {
                'DigitalOcean.Droplet': waldur_digitalocean.models.Droplet,
                'JIRA.Project': waldur_jira.models.Project,
                'OpenStack.Tenant': waldur_openstack.models.Tenant
            }

        """
        from django.apps import apps

        return {'.'.join([service['name'], attrs['name']]): apps.get_model(resource)
                for service in cls._registry.values()
                for resource, attrs in service['resources'].items()}