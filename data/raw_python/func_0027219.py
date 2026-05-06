def list(self, request, *args, **kwargs):
        """
        Project permissions expresses connection of user to a project.
        User may have either project manager or system administrator permission in the project.
        Use */api/project-permissions/* endpoint to maintain project permissions.

        Note that project permissions can be viewed and modified only by customer owners and staff users.

        To list all visible permissions, run a **GET** query against a list.
        Response will contain a list of project users and their brief data.

        To add a new user to the project, **POST** a new relationship to */api/project-permissions/* endpoint specifying
        project, user and the role of the user ('admin' or 'manager'):

        .. code-block:: http

            POST /api/project-permissions/ HTTP/1.1
            Accept: application/json
            Authorization: Token 95a688962bf68678fd4c8cec4d138ddd9493c93b
            Host: example.com

            {
                "project": "http://example.com/api/projects/6c9b01c251c24174a6691a1f894fae31/",
                "role": "manager",
                "user": "http://example.com/api/users/82cec6c8e0484e0ab1429412fe4194b7/"
            }
        """
        return super(ProjectPermissionViewSet, self).list(request, *args, **kwargs)