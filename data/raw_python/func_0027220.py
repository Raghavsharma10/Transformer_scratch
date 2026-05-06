def destroy(self, request, *args, **kwargs):
        """
        To remove a user from a project, delete corresponding connection (**url** field). Successful deletion
        will return status code 204.

        .. code-block:: http

            DELETE /api/project-permissions/42/ HTTP/1.1
            Authorization: Token 95a688962bf68678fd4c8cec4d138ddd9493c93b
            Host: example.com
        """
        return super(ProjectPermissionViewSet, self).destroy(request, *args, **kwargs)