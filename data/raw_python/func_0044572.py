def get_relationship(cls, request_args, id, related_collection_name, related_resource=None):
        """
        Get a relationship
        :param request_args:
        :param id: The 'id' field of the node on the left side of the relationship in the database.  The id field must \
        be set in the model -- it is not the same as the node id
        :param related_collection_name: The name of the relationship
        :param related_resource: Deprecated for version 1.1.0
        :return: A response according to the specification at http://jsonapi.org/format/#fetching-relationships
        """
        try:
            included = request_args.get('include').split(',')
        except (SyntaxError, AttributeError):
            included = []
        try:
            offset = request_args.get('page[offset]', 0)
            limit = request_args.get('page[limit]', 20)
            this_resource = cls.nodes.get(id=id, active=True)
            if not related_resource:
                if request_args.get('include'):
                    r = application_codes.error_response([application_codes.PARAMETER_NOT_SUPPORTED_VIOLATION])
                else:
                    r = this_resource.relationship_collection_response(related_collection_name, offset, limit)
            else:  # deprecated for version 1.1.0
                r = this_resource.individual_relationship_response(related_collection_name, related_resource, included)

        except DoesNotExist:
            r = application_codes.error_response([application_codes.RESOURCE_NOT_FOUND])
        return r