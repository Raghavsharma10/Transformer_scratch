def deactivate_resource(cls, id):
        r"""
        Used to deactivate a node of type 'cls' in response to a DELETE request. deactivate_resource should only \
        be invoked on a resource when the client specifies a DELETE request.

        :param id: The 'id' field of the node to update in the database.  The id field must be set in the model -- it \
        is not the same as the node id
        :return: An HTTP response object in accordance with the specification at \
        http://jsonapi.org/format/#crud-deleting
        """
        try:
            this_resource = cls.nodes.get(id=id, active=True)
            this_resource.deactivate()
            r = make_response('')
            r.headers['Content-Type'] = "application/vnd.api+json; charset=utf-8"
            r.status_code = http_error_codes.NO_CONTENT
        except DoesNotExist:
            r = application_codes.error_response([application_codes.RESOURCE_NOT_FOUND])

        return r