def list(self, request, *args, **kwargs):
        """
        Each customer is associated with a group of users that represent customer owners. The link is maintained
        through **api/customer-permissions/** endpoint.

        To list all visible links, run a **GET** query against a list.
        Response will contain a list of customer owners and their brief data.

        To add a new user to the customer, **POST** a new relationship to **customer-permissions** endpoint:

        .. code-block:: http

            POST /api/customer-permissions/ HTTP/1.1
            Accept: application/json
            Authorization: Token 95a688962bf68678fd4c8cec4d138ddd9493c93b
            Host: example.com

            {
                "customer": "http://example.com/api/customers/6c9b01c251c24174a6691a1f894fae31/",
                "role": "owner",
                "user": "http://example.com/api/users/82cec6c8e0484e0ab1429412fe4194b7/"
            }
        """
        return super(CustomerPermissionViewSet, self).list(request, *args, **kwargs)