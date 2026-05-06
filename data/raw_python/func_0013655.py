def describe_events(SourceIdentifier=None, SourceType=None, StartTime=None, EndTime=None, Duration=None, EventCategories=None, Filters=None, MaxRecords=None, Marker=None):
    """
    Lists events for a given source identifier and source type. You can also specify a start and end time. For more information on AWS DMS events, see Working with Events and Notifications .
    See also: AWS API Documentation
    
    
    :example: response = client.describe_events(
        SourceIdentifier='string',
        SourceType='replication-instance',
        StartTime=datetime(2015, 1, 1),
        EndTime=datetime(2015, 1, 1),
        Duration=123,
        EventCategories=[
            'string',
        ],
        Filters=[
            {
                'Name': 'string',
                'Values': [
                    'string',
                ]
            },
        ],
        MaxRecords=123,
        Marker='string'
    )
    
    
    :type SourceIdentifier: string
    :param SourceIdentifier: The identifier of the event source. An identifier must begin with a letter and must contain only ASCII letters, digits, and hyphens. It cannot end with a hyphen or contain two consecutive hyphens.

    :type SourceType: string
    :param SourceType: The type of AWS DMS resource that generates events.
            Valid values: replication-instance | migration-task
            

    :type StartTime: datetime
    :param StartTime: The start time for the events to be listed.

    :type EndTime: datetime
    :param EndTime: The end time for the events to be listed.

    :type Duration: integer
    :param Duration: The duration of the events to be listed.

    :type EventCategories: list
    :param EventCategories: A list of event categories for a source type that you want to subscribe to.
            (string) --
            

    :type Filters: list
    :param Filters: Filters applied to the action.
            (dict) --
            Name (string) -- [REQUIRED]The name of the filter.
            Values (list) -- [REQUIRED]The filter value.
            (string) --
            
            

    :type MaxRecords: integer
    :param MaxRecords: The maximum number of records to include in the response. If more records exist than the specified MaxRecords value, a pagination token called a marker is included in the response so that the remaining results can be retrieved.
            Default: 100
            Constraints: Minimum 20, maximum 100.
            

    :type Marker: string
    :param Marker: An optional pagination token provided by a previous request. If this parameter is specified, the response includes only records beyond the marker, up to the value specified by MaxRecords .

    :rtype: dict
    :return: {
        'Marker': 'string',
        'Events': [
            {
                'SourceIdentifier': 'string',
                'SourceType': 'replication-instance',
                'Message': 'string',
                'EventCategories': [
                    'string',
                ],
                'Date': datetime(2015, 1, 1)
            },
        ]
    }
    
    
    :returns: 
    (string) --
    
    """
    pass