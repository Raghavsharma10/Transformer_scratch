def get_items_by_query(self, item_query):
        """Gets a list of ``Items`` matching the given item query.

        arg:    item_query (osid.assessment.ItemQuery): the item query
        return: (osid.assessment.ItemList) - the returned ``ItemList``
        raise:  NullArgument - ``item_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``item_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in item_query._query_terms:
            if '$in' in item_query._query_terms[term] and '$nin' in item_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': item_query._query_terms[term]['$in']}},
                             {term: {'$nin': item_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: item_query._query_terms[term]})
        for term in item_query._keyword_terms:
            or_list.append({term: item_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('assessment',
                                             collection='Item',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.ItemList(result, runtime=self._runtime, proxy=self._proxy)