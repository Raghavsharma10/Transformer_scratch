def get_resource_or_collection(cls, request_args, id=None):
        r"""
        Deprecated for version 1.1.0. Please use get_resource or get_collection.

        This function has multiple behaviors.

        With id specified: Used to fetch a single resource object with the given id in response to a GET request.\
        get_resource_or_collection should only be invoked on a resource when the client specifies a GET request.

        With id not specified: Used to fetch a collection of resource object of type 'cls' in response to a GET request\
        . get_resource_or_collection should only be invoked on a resource when the client specifies a GET request.

        :param request_args: The query parameters supplied with the request.  currently supports include, page[offset], \
        and page[limit]. Pagination only applies to collection requests. See http://jsonapi.org/format/#fetching-pagination and \
        http://jsonapi.org/format/#fetching-includes
        :param id: The 'id' field of the node to fetch in the database.  The id field must be set in the model -- it \
        is not the same as the node id.  If the id is not supplied the full collection will be returned.
        :return: An HTTP response object in accordance with the specification at \
        http://jsonapi.org/format/#fetching-resources
        """
        if id:
            try:
                r = cls.get_resource(request_args)
            except DoesNotExist:
                r = application_codes.error_response([application_codes.RESOURCE_NOT_FOUND])
        else:
            try:
                r = cls.get_collection(request_args)
            except Exception as e:
                r = application_codes.error_response([application_codes.BAD_FORMAT_VIOLATION])
        return r