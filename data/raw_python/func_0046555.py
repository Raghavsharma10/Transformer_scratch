def get_relationships_by_query(self, relationship_query):
        """Gets a list of ``Relationships`` matching the given relationship query.

        arg:    relationship_query
                (osid.relationship.RelationshipQuery): the relationship
                query
        return: (osid.relationship.RelationshipList) - the returned
                ``RelationshipList``
        raise:  NullArgument - ``relationship_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``relationship_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in relationship_query._query_terms:
            if '$in' in relationship_query._query_terms[term] and '$nin' in relationship_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': relationship_query._query_terms[term]['$in']}},
                             {term: {'$nin': relationship_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: relationship_query._query_terms[term]})
        for term in relationship_query._keyword_terms:
            or_list.append({term: relationship_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('relationship',
                                             collection='Relationship',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.RelationshipList(result, runtime=self._runtime, proxy=self._proxy)