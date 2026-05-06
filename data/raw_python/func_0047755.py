def get_resources_by_search(self, resource_query, resource_search):
        """Gets the search results matching the given search query using the given search.

        arg:    resource_query (osid.resource.ResourceQuery): the
                resource query
        arg:    resource_search (osid.resource.ResourceSearch): the
                resource search
        return: (osid.resource.ResourceSearchResults) - the resource
                search results
        raise:  NullArgument - ``resource_query`` or ``resource_search``
                is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``resource_query`` or ``resource_search``
                is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        # Copied from osid.resource.ResourceQuerySession.get_resources_by_query_template
        and_list = list()
        or_list = list()
        for term in resource_query._query_terms:
            and_list.append({term: resource_query._query_terms[term]})
        for term in resource_query._keyword_terms:
            or_list.append({term: resource_query._keyword_terms[term]})
        if resource_search._id_list is not None:
            identifiers = [ObjectId(i.identifier) for i in resource_search._id_list]
            and_list.append({'_id': {'$in': identifiers}})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        if resource_search.start is not None and resource_search.end is not None:
            result = collection.find(query_terms)[resource_search.start:resource_search.end]
        else:
            result = collection.find(query_terms)
        return searches.ResourceSearchResults(result, dict(resource_query._query_terms), runtime=self._runtime)