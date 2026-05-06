def get_resources(cls, request=None):
        """ Get a list of resources endpoints.
            {
                "DigitalOcean.Droplet": "/api/digitalocean-droplets/",
                "Oracle.Database": "/api/oracle-databases/",
                "GitLab.Group": "/api/gitlab-groups/",
                "GitLab.Project": "/api/gitlab-projects/"
            }
        """
        return {'.'.join([service['name'], resource['name']]): reverse(resource['list_view'], request=request)
                for service in cls._registry.values()
                for resource in service['resources'].values()}