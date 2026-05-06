def disconnect_relationship(cls, id, related_collection_name, request_json):
        """
        Disconnect one or more relationship in a collection with cardinality 'Many'.

        :param id: The 'id' field of the node on the left side of the relationship in the database.  The id field must \
        be set in the model -- it is not the same as the node id
        :param related_collection_name: The name of the relationship
        :param request_json: a dictionary formatted according to the specification at \
        http://jsonapi.org/format/#crud-updating-relationships
        :return: A response according to the same specification
        """
        try:
            this_resource = cls.nodes.get(id=id, active=True)
            related_collection = getattr(this_resource, related_collection_name)
            rsrc_identifier_list = request_json['data']
            if not isinstance(rsrc_identifier_list, list):
                raise WrongTypeError

            for rsrc_identifier in rsrc_identifier_list:
                connected_resource = cls.get_class_from_type(rsrc_identifier['type']).nodes.get(
                    id=rsrc_identifier['id']
                )
                related_collection.disconnect(connected_resource)
            r = make_response('')
            r.status_code = http_error_codes.NO_CONTENT
            r.headers['Content-Type'] = CONTENT_TYPE
        except DoesNotExist:
            r = application_codes.error_response([application_codes.RESOURCE_NOT_FOUND])
        except (KeyError, WrongTypeError):
            r = application_codes.error_response([application_codes.BAD_FORMAT_VIOLATION])
        return r