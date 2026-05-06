def list(self, request, *args, **kwargs):
        """
        To get a list of projects, run **GET** against */api/projects/* as authenticated user.
        Here you can also check actual value for project quotas and project usage

        Note that a user can only see connected projects:

        - projects that the user owns as a customer
        - projects where user has any role

        Supported logic filters:

        - ?can_manage - return a list of projects where current user is manager or a customer owner;
        - ?can_admin - return a list of projects where current user is admin;
        """
        return super(ProjectViewSet, self).list(request, *args, **kwargs)