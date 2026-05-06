def retrieve(self, request, *args, **kwargs):
        """
        To set quota limit issue a **PUT** request against */api/quotas/<quota uuid>** with limit values.

        Please note that if a quota is a cache of a backend quota (e.g. 'storage' size of an OpenStack tenant),
        it will be impossible to modify it through */api/quotas/<quota uuid>** endpoint.

        Example of changing quota limit:

        .. code-block:: http

            POST /api/quotas/6ad5f49d6d6c49648573b2b71f44a42b/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "limit": 2000.0
            }

        Example of changing quota threshold:

        .. code-block:: http

            PUT /api/quotas/6ad5f49d6d6c49648573b2b71f44a42b/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "threshold": 100.0
            }

        """
        return super(QuotaViewSet, self).retrieve(request, *args, **kwargs)