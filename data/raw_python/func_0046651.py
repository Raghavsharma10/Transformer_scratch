def get_authorizations_by_query(self, authorization_query):
        """Gets a list of ``Authorizations`` matching the given query.

        arg:    authorization_query
                (osid.authorization.AuthorizationQuery): the
                authorization query
        return: (osid.authorization.AuthorizationList) - the returned
                ``AuthorizationList``
        raise:  NullArgument - ``authorization_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``authorization_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in authorization_query._query_terms:
            if '$in' in authorization_query._query_terms[term] and '$nin' in authorization_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': authorization_query._query_terms[term]['$in']}},
                             {term: {'$nin': authorization_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: authorization_query._query_terms[term]})
        for term in authorization_query._keyword_terms:
            or_list.append({term: authorization_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('authorization',
                                             collection='Authorization',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.AuthorizationList(result, runtime=self._runtime, proxy=self._proxy)