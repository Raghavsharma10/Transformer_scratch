def update_relationship(cls, id, related_collection_name, request_json):
        r"""
        Used to completely replace all the existing relationships with new ones.

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
            data = request_json['data']

            if type(related_collection) in (One, ZeroOrOne):  # Cardinality <= 1 so is a single obj
                if not data and related_collection.single():  # disconnect the resource
                    related_collection.disconnect(related_collection.single())
                elif not data:
                    pass  # There is already no connected resource
                else:
                    the_new_node = cls.get_class_from_type(data['type']).nodes.get(id=data['id'])
                    if related_collection.single():  # update the relationship
                        related_collection.reconnect(related_collection.single(), the_new_node)
                        the_rel = eval('related_collection.relationship(the_new_node)'.format(
                            start_node=this_resource, relname=related_collection_name)
                        )
                        meta = data.get('meta')
                        if meta:
                            for k in meta.keys():
                                setattr(the_rel, k, meta[k])
                        the_rel.save()

                    else:  # create the relationship
                        related_collection.connect(the_new_node, data.get('meta'))

            else:  # Cardinality > 1 so this is a collection of objects
                old_nodes = related_collection.all()
                for item in old_nodes:  # removes all old connections
                    related_collection.disconnect(item)
                for identifier in data:  # adds all new connections
                    the_new_node = cls.get_class_from_type(identifier['type']).nodes.get(id=identifier['id'])
                    the_rel = related_collection.connect(the_new_node)
                    meta = identifier.get('meta')
                    if meta:
                        for k in meta.keys():
                            setattr(the_rel, k, meta[k])
                    the_rel.save()

            r = make_response('')
            r.status_code = http_error_codes.NO_CONTENT
            r.headers['Content-Type'] = CONTENT_TYPE

        except DoesNotExist:
            r = application_codes.error_response([application_codes.RESOURCE_NOT_FOUND])
        except (KeyError, TypeError):
            r = application_codes.error_response([application_codes.BAD_FORMAT_VIOLATION])
        except AttemptedCardinalityViolation:
            r = application_codes.error_response([application_codes.ATTEMPTED_CARDINALITY_VIOLATION])
        except MultipleNodesReturned:
            r = application_codes.error_response([application_codes.MULTIPLE_NODES_WITH_ID_VIOLATION])
        return r