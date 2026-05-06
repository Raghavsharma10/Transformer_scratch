def create_replication_task(ReplicationTaskIdentifier=None, SourceEndpointArn=None, TargetEndpointArn=None, ReplicationInstanceArn=None, MigrationType=None, TableMappings=None, ReplicationTaskSettings=None, CdcStartTime=None, Tags=None):
    """
    Creates a replication task using the specified parameters.
    See also: AWS API Documentation
    
    
    :example: response = client.create_replication_task(
        ReplicationTaskIdentifier='string',
        SourceEndpointArn='string',
        TargetEndpointArn='string',
        ReplicationInstanceArn='string',
        MigrationType='full-load'|'cdc'|'full-load-and-cdc',
        TableMappings='string',
        ReplicationTaskSettings='string',
        CdcStartTime=datetime(2015, 1, 1),
        Tags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ]
    )
    
    
    :type ReplicationTaskIdentifier: string
    :param ReplicationTaskIdentifier: [REQUIRED]
            The replication task identifier.
            Constraints:
            Must contain from 1 to 255 alphanumeric characters or hyphens.
            First character must be a letter.
            Cannot end with a hyphen or contain two consecutive hyphens.
            

    :type SourceEndpointArn: string
    :param SourceEndpointArn: [REQUIRED]
            The Amazon Resource Name (ARN) string that uniquely identifies the endpoint.
            

    :type TargetEndpointArn: string
    :param TargetEndpointArn: [REQUIRED]
            The Amazon Resource Name (ARN) string that uniquely identifies the endpoint.
            

    :type ReplicationInstanceArn: string
    :param ReplicationInstanceArn: [REQUIRED]
            The Amazon Resource Name (ARN) of the replication instance.
            

    :type MigrationType: string
    :param MigrationType: [REQUIRED]
            The migration type.
            

    :type TableMappings: string
    :param TableMappings: [REQUIRED]
            When using the AWS CLI or boto3, provide the path of the JSON file that contains the table mappings. Precede the path with 'file://'. When working with the DMS API, provide the JSON as the parameter value.
            For example, --table-mappings file://mappingfile.json
            

    :type ReplicationTaskSettings: string
    :param ReplicationTaskSettings: Settings for the task, such as target metadata settings. For a complete list of task settings, see Task Settings for AWS Database Migration Service Tasks .

    :type CdcStartTime: datetime
    :param CdcStartTime: The start time for the Change Data Capture (CDC) operation.

    :type Tags: list
    :param Tags: Tags to be added to the replication instance.
            (dict) --
            Key (string) --A key is the required name of the tag. The string value can be from 1 to 128 Unicode characters in length and cannot be prefixed with 'aws:' or 'dms:'. The string can only contain only the set of Unicode letters, digits, white-space, '_', '.', '/', '=', '+', '-' (Java regex: '^([\p{L}\p{Z}\p{N}_.:/=+\-]*)$').
            Value (string) --A value is the optional value of the tag. The string value can be from 1 to 256 Unicode characters in length and cannot be prefixed with 'aws:' or 'dms:'. The string can only contain only the set of Unicode letters, digits, white-space, '_', '.', '/', '=', '+', '-' (Java regex: '^([\p{L}\p{Z}\p{N}_.:/=+\-]*)$').
            
            

    :rtype: dict
    :return: {
        'ReplicationTask': {
            'ReplicationTaskIdentifier': 'string',
            'SourceEndpointArn': 'string',
            'TargetEndpointArn': 'string',
            'ReplicationInstanceArn': 'string',
            'MigrationType': 'full-load'|'cdc'|'full-load-and-cdc',
            'TableMappings': 'string',
            'ReplicationTaskSettings': 'string',
            'Status': 'string',
            'LastFailureMessage': 'string',
            'StopReason': 'string',
            'ReplicationTaskCreationDate': datetime(2015, 1, 1),
            'ReplicationTaskStartDate': datetime(2015, 1, 1),
            'ReplicationTaskArn': 'string',
            'ReplicationTaskStats': {
                'FullLoadProgressPercent': 123,
                'ElapsedTimeMillis': 123,
                'TablesLoaded': 123,
                'TablesLoading': 123,
                'TablesQueued': 123,
                'TablesErrored': 123
            }
        }
    }
    
    
    :returns: 
    Must contain from 1 to 255 alphanumeric characters or hyphens.
    First character must be a letter.
    Cannot end with a hyphen or contain two consecutive hyphens.
    
    """
    pass