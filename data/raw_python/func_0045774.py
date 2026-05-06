def get_log_entries_by_query(self, log_entry_query):
        """Gets a list of log entries matching the given log entry query.

        arg:    log_entry_query (osid.logging.LogEntryQuery): the log
                entry query
        return: (osid.logging.LogEntryList) - the returned
                ``LogEntryList``
        raise:  NullArgument - ``log_entry_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``log_entry_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in log_entry_query._query_terms:
            if '$in' in log_entry_query._query_terms[term] and '$nin' in log_entry_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': log_entry_query._query_terms[term]['$in']}},
                             {term: {'$nin': log_entry_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: log_entry_query._query_terms[term]})
        for term in log_entry_query._keyword_terms:
            or_list.append({term: log_entry_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('logging',
                                             collection='LogEntry',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.LogEntryList(result, runtime=self._runtime, proxy=self._proxy)