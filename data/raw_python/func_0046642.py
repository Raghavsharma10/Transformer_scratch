def is_authorized(self, agent_id, function_id, qualifier_id):
        """Determines if the given agent is authorized.

        An agent is authorized if an active authorization exists whose
        ``Agent,`` ``Function`` and ``Qualifier`` matches the supplied
        parameters. Authorizations may be defined using groupings or
        hieratchical structures for both the ``Agent`` and the
        ``Qualifier`` but are queried in the de-nornmalized form.

        The ``Agent`` is generally determined through the use of an
        Authentication OSID. The ``Function`` and ``Qualifier`` are
        already known as they map to the desired authorization to
        validate.

        arg:    agent_id (osid.id.Id): the ``Id`` of an ``Agent``
        arg:    function_id (osid.id.Id): the ``Id`` of a ``Function``
        arg:    qualifier_id (osid.id.Id): the ``Id`` of a ``Qualifier``
        return: (boolean) - ``true`` if the user is authorized,
                ``false`` othersise
        raise:  NotFound - ``function_id`` is not found
        raise:  NullArgument - ``agent_id`` , ``function_id`` or
                ``qualifier_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure making request
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: Authorizations may be stored in a
        normalized form with respect to various Resources and created
        using specific nodes in a ``Function`` or ``Qualifer``
        hierarchy. The provider needs to maintain a de-normalized
        implicit authorization store or expand the applicable
        hierarchies on the fly to honor this query.  Querying the
        authorization service may in itself require a separate
        authorization. A ``PermissionDenied`` is a result of this
        authorization failure. If no explicit or implicit authorization
        exists for the queried tuple, this method should return
        ``false``.

        """
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)

        def is_parent_authorized(catalog_id):
            """Recursively checks parents for implicit authorizations"""
            parent_id_list = self._get_parent_id_list(catalog_id, hierarchy_id)
            if parent_id_list:
                try:
                    collection.find_one(
                        {'agentId': str(agent_id),
                         'functionId': str(function_id),
                         'qualifierId': {'$in': parent_id_list}})
                except errors.NotFound:
                    for parent_id in parent_id_list:
                        if is_parent_authorized(Id(parent_id)):
                            return True
                    return False
                else:
                    return True
            else:
                return False

        # Check first for an explicit or 'ROOT' level implicit authorization:
        try:
            authority = qualifier_id.get_identifier_namespace().split('.')[0].upper()
            identifier = qualifier_id.get_identifier_namespace().split('.')[1].upper()
        except KeyError:
            idstr_list = [str(qualifier_id)]
            authority = identifier = None
        else:
            # handle aliased IDs
            package_name = qualifier_id.get_identifier_namespace().split('.')[0]
            qualifier_id = self._get_id(qualifier_id, package_name)

            root_qualifier_id = Id(
                authority=qualifier_id.get_authority(),
                namespace=qualifier_id.get_identifier_namespace(),
                identifier='ROOT')
            idstr_list = [str(root_qualifier_id), str(qualifier_id)]
        try:
            collection.find_one(
                {'agentId': str(agent_id),
                 'functionId': str(function_id),
                 'qualifierId': {'$in': idstr_list}})

        # Otherwise check for implicit authorization through inheritance:
        except errors.NotFound:
            if authority and identifier:
                hierarchy_id = Id(authority=authority,
                                  namespace='CATALOG',
                                  identifier=identifier)
                return is_parent_authorized(qualifier_id)
            else:
                return False
        else:
            return True