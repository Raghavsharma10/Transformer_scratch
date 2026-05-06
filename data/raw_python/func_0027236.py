def list(self, request, *args, **kwargs):
        """
        To get a list of connections between a project and an service, run **GET** against service_project_link_url
        as authenticated user. Note that a user can only see connections of a project where a user has a role.

        If service has `available_for_all` flag, project-service connections are created automatically.
        Otherwise, in order to be able to provision resources, service must first be linked to a project.
        To do that, **POST** a connection between project and a service to service_project_link_url
        as stuff user or customer owner.
        """
        return super(BaseServiceProjectLinkViewSet, self).list(request, *args, **kwargs)