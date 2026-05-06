def link(self, request, uuid=None):
        """
        To get a list of resources available for import, run **GET** against */<service_endpoint>/link/*
        as an authenticated user.
        Optionally project_uuid parameter can be supplied for services requiring it like OpenStack.

        To import (link with Waldur) resource issue **POST** against the same endpoint with resource id.

        .. code-block:: http

            POST /api/openstack/08039f01c9794efc912f1689f4530cf0/link/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "backend_id": "bd5ec24d-9164-440b-a9f2-1b3c807c5df3",
                "project": "http://example.com/api/projects/e5f973af2eb14d2d8c38d62bcbaccb33/"
            }
        """

        service = self.get_object()

        if self.request.method == 'GET':
            try:
                backend = self.get_backend(service)
                try:
                    resources = backend.get_resources_for_import(**self.get_import_context())
                except ServiceBackendNotImplemented:
                    resources = []

                page = self.paginate_queryset(resources)
                if page is not None:
                    return self.get_paginated_response(page)

                return Response(resources)
            except (ServiceBackendError, ValidationError) as e:
                raise APIException(e)

        else:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            try:
                resource = serializer.save()
            except ServiceBackendError as e:
                raise APIException(e)

            resource_imported.send(
                sender=resource.__class__,
                instance=resource,
            )

            return Response(serializer.data, status=status.HTTP_200_OK)