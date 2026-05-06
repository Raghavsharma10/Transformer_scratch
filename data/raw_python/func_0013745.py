def list_closed_workflow_executions(domain=None, startTimeFilter=None, closeTimeFilter=None, executionFilter=None, closeStatusFilter=None, typeFilter=None, tagFilter=None, nextPageToken=None, maximumPageSize=None, reverseOrder=None):
    """
    Returns a list of closed workflow executions in the specified domain that meet the filtering criteria. The results may be split into multiple pages. To retrieve subsequent pages, make the call again using the nextPageToken returned by the initial call.
    Access Control
    You can use IAM policies to control this action's access to Amazon SWF resources as follows:
    If the caller does not have sufficient permissions to invoke the action, or the parameter values fall outside the specified constraints, the action fails. The associated event attribute's cause parameter will be set to OPERATION_NOT_PERMITTED. For details and example IAM policies, see Using IAM to Manage Access to Amazon SWF Workflows .
    See also: AWS API Documentation
    
    
    :example: response = client.list_closed_workflow_executions(
        domain='string',
        startTimeFilter={
            'oldestDate': datetime(2015, 1, 1),
            'latestDate': datetime(2015, 1, 1)
        },
        closeTimeFilter={
            'oldestDate': datetime(2015, 1, 1),
            'latestDate': datetime(2015, 1, 1)
        },
        executionFilter={
            'workflowId': 'string'
        },
        closeStatusFilter={
            'status': 'COMPLETED'|'FAILED'|'CANCELED'|'TERMINATED'|'CONTINUED_AS_NEW'|'TIMED_OUT'
        },
        typeFilter={
            'name': 'string',
            'version': 'string'
        },
        tagFilter={
            'tag': 'string'
        },
        nextPageToken='string',
        maximumPageSize=123,
        reverseOrder=True|False
    )
    
    
    :type domain: string
    :param domain: [REQUIRED]
            The name of the domain that contains the workflow executions to list.
            

    :type startTimeFilter: dict
    :param startTimeFilter: If specified, the workflow executions are included in the returned results based on whether their start times are within the range specified by this filter. Also, if this parameter is specified, the returned results are ordered by their start times.
            Note
            startTimeFilter and closeTimeFilter are mutually exclusive. You must specify one of these in a request but not both.
            oldestDate (datetime) -- [REQUIRED]Specifies the oldest start or close date and time to return.
            latestDate (datetime) --Specifies the latest start or close date and time to return.
            

    :type closeTimeFilter: dict
    :param closeTimeFilter: If specified, the workflow executions are included in the returned results based on whether their close times are within the range specified by this filter. Also, if this parameter is specified, the returned results are ordered by their close times.
            Note
            startTimeFilter and closeTimeFilter are mutually exclusive. You must specify one of these in a request but not both.
            oldestDate (datetime) -- [REQUIRED]Specifies the oldest start or close date and time to return.
            latestDate (datetime) --Specifies the latest start or close date and time to return.
            

    :type executionFilter: dict
    :param executionFilter: If specified, only workflow executions matching the workflow ID specified in the filter are returned.
            Note
            closeStatusFilter , executionFilter , typeFilter and tagFilter are mutually exclusive. You can specify at most one of these in a request.
            workflowId (string) -- [REQUIRED]The workflowId to pass of match the criteria of this filter.
            

    :type closeStatusFilter: dict
    :param closeStatusFilter: If specified, only workflow executions that match this close status are listed. For example, if TERMINATED is specified, then only TERMINATED workflow executions are listed.
            Note
            closeStatusFilter , executionFilter , typeFilter and tagFilter are mutually exclusive. You can specify at most one of these in a request.
            status (string) -- [REQUIRED]Required. The close status that must match the close status of an execution for it to meet the criteria of this filter.
            

    :type typeFilter: dict
    :param typeFilter: If specified, only executions of the type specified in the filter are returned.
            Note
            closeStatusFilter , executionFilter , typeFilter and tagFilter are mutually exclusive. You can specify at most one of these in a request.
            name (string) -- [REQUIRED]Required. Name of the workflow type.
            version (string) --Version of the workflow type.
            

    :type tagFilter: dict
    :param tagFilter: If specified, only executions that have the matching tag are listed.
            Note
            closeStatusFilter , executionFilter , typeFilter and tagFilter are mutually exclusive. You can specify at most one of these in a request.
            tag (string) -- [REQUIRED]Required. Specifies the tag that must be associated with the execution for it to meet the filter criteria.
            

    :type nextPageToken: string
    :param nextPageToken: If a NextPageToken was returned by a previous call, there are more results available. To retrieve the next page of results, make the call again using the returned token in nextPageToken . Keep all other arguments unchanged.
            The configured maximumPageSize determines how many results can be returned in a single call.
            

    :type maximumPageSize: integer
    :param maximumPageSize: The maximum number of results that will be returned per call. nextPageToken can be used to obtain futher pages of results. The default is 1000, which is the maximum allowed page size. You can, however, specify a page size smaller than the maximum.
            This is an upper limit only; the actual number of results returned per call may be fewer than the specified maximum.
            

    :type reverseOrder: boolean
    :param reverseOrder: When set to true , returns the results in reverse order. By default the results are returned in descending order of the start or the close time of the executions.

    :rtype: dict
    :return: {
        'executionInfos': [
            {
                'execution': {
                    'workflowId': 'string',
                    'runId': 'string'
                },
                'workflowType': {
                    'name': 'string',
                    'version': 'string'
                },
                'startTimestamp': datetime(2015, 1, 1),
                'closeTimestamp': datetime(2015, 1, 1),
                'executionStatus': 'OPEN'|'CLOSED',
                'closeStatus': 'COMPLETED'|'FAILED'|'CANCELED'|'TERMINATED'|'CONTINUED_AS_NEW'|'TIMED_OUT',
                'parent': {
                    'workflowId': 'string',
                    'runId': 'string'
                },
                'tagList': [
                    'string',
                ],
                'cancelRequested': True|False
            },
        ],
        'nextPageToken': 'string'
    }
    
    
    :returns: 
    domain (string) -- [REQUIRED]
    The name of the domain that contains the workflow executions to list.
    
    startTimeFilter (dict) -- If specified, the workflow executions are included in the returned results based on whether their start times are within the range specified by this filter. Also, if this parameter is specified, the returned results are ordered by their start times.
    
    Note
    startTimeFilter and closeTimeFilter are mutually exclusive. You must specify one of these in a request but not both.
    
    
    oldestDate (datetime) -- [REQUIRED]Specifies the oldest start or close date and time to return.
    
    latestDate (datetime) --Specifies the latest start or close date and time to return.
    
    
    
    closeTimeFilter (dict) -- If specified, the workflow executions are included in the returned results based on whether their close times are within the range specified by this filter. Also, if this parameter is specified, the returned results are ordered by their close times.
    
    Note
    startTimeFilter and closeTimeFilter are mutually exclusive. You must specify one of these in a request but not both.
    
    
    oldestDate (datetime) -- [REQUIRED]Specifies the oldest start or close date and time to return.
    
    latestDate (datetime) --Specifies the latest start or close date and time to return.
    
    
    
    executionFilter (dict) -- If specified, only workflow executions matching the workflow ID specified in the filter are returned.
    
    Note
    closeStatusFilter , executionFilter , typeFilter and tagFilter are mutually exclusive. You can specify at most one of these in a request.
    
    
    workflowId (string) -- [REQUIRED]The workflowId to pass of match the criteria of this filter.
    
    
    
    closeStatusFilter (dict) -- If specified, only workflow executions that match this close status are listed. For example, if TERMINATED is specified, then only TERMINATED workflow executions are listed.
    
    Note
    closeStatusFilter , executionFilter , typeFilter and tagFilter are mutually exclusive. You can specify at most one of these in a request.
    
    
    status (string) -- [REQUIRED]Required. The close status that must match the close status of an execution for it to meet the criteria of this filter.
    
    
    
    typeFilter (dict) -- If specified, only executions of the type specified in the filter are returned.
    
    Note
    closeStatusFilter , executionFilter , typeFilter and tagFilter are mutually exclusive. You can specify at most one of these in a request.
    
    
    name (string) -- [REQUIRED]Required. Name of the workflow type.
    
    version (string) --Version of the workflow type.
    
    
    
    tagFilter (dict) -- If specified, only executions that have the matching tag are listed.
    
    Note
    closeStatusFilter , executionFilter , typeFilter and tagFilter are mutually exclusive. You can specify at most one of these in a request.
    
    
    tag (string) -- [REQUIRED]Required. Specifies the tag that must be associated with the execution for it to meet the filter criteria.
    
    
    
    nextPageToken (string) -- If a NextPageToken was returned by a previous call, there are more results available. To retrieve the next page of results, make the call again using the returned token in nextPageToken . Keep all other arguments unchanged.
    The configured maximumPageSize determines how many results can be returned in a single call.
    
    maximumPageSize (integer) -- The maximum number of results that will be returned per call. nextPageToken can be used to obtain futher pages of results. The default is 1000, which is the maximum allowed page size. You can, however, specify a page size smaller than the maximum.
    This is an upper limit only; the actual number of results returned per call may be fewer than the specified maximum.
    
    reverseOrder (boolean) -- When set to true , returns the results in reverse order. By default the results are returned in descending order of the start or the close time of the executions.
    
    """
    pass