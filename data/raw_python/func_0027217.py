def retrieve(self, request, *args, **kwargs):
        """
        User fields can be updated by account owner or user with staff privilege (is_staff=True).
        Following user fields can be updated:

        - organization (deprecated, use
          `organization plugin <http://waldur_core-organization.readthedocs.org/en/stable/>`_ instead)
        - full_name
        - native_name
        - job_title
        - phone_number
        - email

        Can be done by **PUT**ing a new data to the user URI, i.e. */api/users/<UUID>/* by staff user or account owner.
        Valid request example (token is user specific):

        .. code-block:: http

            PUT /api/users/e0c058d06864441fb4f1c40dee5dd4fd/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "email": "example@example.com",
                "organization": "Bells organization",
            }
        """
        return super(UserViewSet, self).retrieve(request, *args, **kwargs)