def create_relationships(cls, id, related_collection_name, request_json):
        r"""
        Used to create relationship(s) between the id node and the nodes identified in the included resource \
        identifier objects.

        :param id: The 'id' field of the node on the left side of the relationship in the database.  The id field must \
        be set in the model -- it is not the same as the node id
        :param related_collection_name: The name of the relationship
        :param request_json: request_json: a dictionary formatted according to the specification at \
        http://jsonapi.org/format/#crud-updating-relationships
        :return: A response according to the same specification
        """
        try:
            this_resource = cls.nodes.get(id=id, active=True)
            related_collection = getattr(this_resource, related_collection_name)
            if type(related_collection) in (One, ZeroOrOne):  # Cardinality <= 1 so update_relationship should be used
                r = application_codes.error_response([application_codes.FORBIDDEN_VIOLATION])
            else:
                data = request_json['data']
                for rsrc_identifier in data:
                    the_new_node = cls.get_class_from_type(rsrc_identifier['type']).nodes.get(id=rsrc_identifier['id'])
                    rel_attrs = rsrc_identifier.get('meta')
                    if not rel_attrs or isinstance(rel_attrs, dict):
                        related_collection.connect(the_new_node, rel_attrs)
                    else:
                        raise WrongTypeError
                #r = this_resource.relationship_collection_response(related_collection_name)
                r = make_response('')
                r.status_code = http_error_codes.NO_CONTENT
                r.headers['Content-Type'] = CONTENT_TYPE

        except DoesNotExist:
            r = application_codes.error_response([application_codes.RESOURCE_NOT_FOUND])
        except (KeyError, TypeError, WrongTypeError):
            r = application_codes.error_response([application_codes.BAD_FORMAT_VIOLATION])
        except AttemptedCardinalityViolation:
            r = application_codes.error_response([application_codes.ATTEMPTED_CARDINALITY_VIOLATION])
        except MultipleNodesReturned:
            r = application_codes.error_response([application_codes.MULTIPLE_NODES_WITH_ID_VIOLATION])
        return r