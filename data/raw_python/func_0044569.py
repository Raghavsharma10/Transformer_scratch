def get_resource(cls, request_args, id):
        r"""
        Used to fetch a single resource object with the given id in response to a GET request.\
        get_resource should only be invoked on a resource when the client specifies a GET request.

        :param request_args:
        :return: The query parameters supplied with the request.  currently supports include.  See \
        http://jsonapi.org/format/#fetching-includes
        """
        try:
            this_resource = cls.nodes.get(id=id, active=True)

            try:
                included = request_args.get('include').split(',')
            except AttributeError:
                included = []

            r = this_resource.individual_resource_response(included)
        except DoesNotExist:
            r = application_codes.error_response([application_codes.RESOURCE_NOT_FOUND])

        return r