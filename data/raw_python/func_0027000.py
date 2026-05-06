def get_services(cls, request=None):
        """ Get a list of services endpoints.
            {
                "Oracle": "/api/oracle/",
                "OpenStack": "/api/openstack/",
                "GitLab": "/api/gitlab/",
                "DigitalOcean": "/api/digitalocean/"
            }
        """
        return {service['name']: reverse(service['list_view'], request=request)
                for service in cls._registry.values()}