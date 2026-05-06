def describe_cluster_snapshots(ClusterIdentifier=None, SnapshotIdentifier=None, SnapshotType=None, StartTime=None, EndTime=None, MaxRecords=None, Marker=None, OwnerAccount=None, TagKeys=None, TagValues=None):
    """
    Returns one or more snapshot objects, which contain metadata about your cluster snapshots. By default, this operation returns information about all snapshots of all clusters that are owned by you AWS customer account. No information is returned for snapshots owned by inactive AWS customer accounts.
    If you specify both tag keys and tag values in the same request, Amazon Redshift returns all snapshots that match any combination of the specified keys and values. For example, if you have owner and environment for tag keys, and admin and test for tag values, all snapshots that have any combination of those values are returned. Only snapshots that you own are returned in the response; shared snapshots are not returned with the tag key and tag value request parameters.
    If both tag keys and values are omitted from the request, snapshots are returned regardless of whether they have tag keys or values associated with them.
    See also: AWS API Documentation
    
    
    :example: response = client.describe_cluster_snapshots(
        ClusterIdentifier='string',
        SnapshotIdentifier='string',
        SnapshotType='string',
        StartTime=datetime(2015, 1, 1),
        EndTime=datetime(2015, 1, 1),
        MaxRecords=123,
        Marker='string',
        OwnerAccount='string',
        TagKeys=[
            'string',
        ],
        TagValues=[
            'string',
        ]
    )
    
    
    :type ClusterIdentifier: string
    :param ClusterIdentifier: The identifier of the cluster for which information about snapshots is requested.

    :type SnapshotIdentifier: string
    :param SnapshotIdentifier: The snapshot identifier of the snapshot about which to return information.

    :type SnapshotType: string
    :param SnapshotType: The type of snapshots for which you are requesting information. By default, snapshots of all types are returned.
            Valid Values: automated | manual
            

    :type StartTime: datetime
    :param StartTime: A value that requests only snapshots created at or after the specified time. The time value is specified in ISO 8601 format. For more information about ISO 8601, go to the ISO8601 Wikipedia page.
            Example: 2012-07-16T18:00:00Z
            

    :type EndTime: datetime
    :param EndTime: A time value that requests only snapshots created at or before the specified time. The time value is specified in ISO 8601 format. For more information about ISO 8601, go to the ISO8601 Wikipedia page.
            Example: 2012-07-16T18:00:00Z
            

    :type MaxRecords: integer
    :param MaxRecords: The maximum number of response records to return in each call. If the number of remaining response records exceeds the specified MaxRecords value, a value is returned in a marker field of the response. You can retrieve the next set of records by retrying the command with the returned marker value.
            Default: 100
            Constraints: minimum 20, maximum 100.
            

    :type Marker: string
    :param Marker: An optional parameter that specifies the starting point to return a set of response records. When the results of a DescribeClusterSnapshots request exceed the value specified in MaxRecords , AWS returns a value in the Marker field of the response. You can retrieve the next set of response records by providing the returned marker value in the Marker parameter and retrying the request.

    :type OwnerAccount: string
    :param OwnerAccount: The AWS customer account used to create or copy the snapshot. Use this field to filter the results to snapshots owned by a particular account. To describe snapshots you own, either specify your AWS customer account, or do not specify the parameter.

    :type TagKeys: list
    :param TagKeys: A tag key or keys for which you want to return all matching cluster snapshots that are associated with the specified key or keys. For example, suppose that you have snapshots that are tagged with keys called owner and environment . If you specify both of these tag keys in the request, Amazon Redshift returns a response with the snapshots that have either or both of these tag keys associated with them.
            (string) --
            

    :type TagValues: list
    :param TagValues: A tag value or values for which you want to return all matching cluster snapshots that are associated with the specified tag value or values. For example, suppose that you have snapshots that are tagged with values called admin and test . If you specify both of these tag values in the request, Amazon Redshift returns a response with the snapshots that have either or both of these tag values associated with them.
            (string) --
            

    :rtype: dict
    :return: {
        'Marker': 'string',
        'Snapshots': [
            {
                'SnapshotIdentifier': 'string',
                'ClusterIdentifier': 'string',
                'SnapshotCreateTime': datetime(2015, 1, 1),
                'Status': 'string',
                'Port': 123,
                'AvailabilityZone': 'string',
                'ClusterCreateTime': datetime(2015, 1, 1),
                'MasterUsername': 'string',
                'ClusterVersion': 'string',
                'SnapshotType': 'string',
                'NodeType': 'string',
                'NumberOfNodes': 123,
                'DBName': 'string',
                'VpcId': 'string',
                'Encrypted': True|False,
                'KmsKeyId': 'string',
                'EncryptedWithHSM': True|False,
                'AccountsWithRestoreAccess': [
                    {
                        'AccountId': 'string',
                        'AccountAlias': 'string'
                    },
                ],
                'OwnerAccount': 'string',
                'TotalBackupSizeInMegaBytes': 123.0,
                'ActualIncrementalBackupSizeInMegaBytes': 123.0,
                'BackupProgressInMegaBytes': 123.0,
                'CurrentBackupRateInMegaBytesPerSecond': 123.0,
                'EstimatedSecondsToCompletion': 123,
                'ElapsedTimeInSeconds': 123,
                'SourceRegion': 'string',
                'Tags': [
                    {
                        'Key': 'string',
                        'Value': 'string'
                    },
                ],
                'RestorableNodeTypes': [
                    'string',
                ],
                'EnhancedVpcRouting': True|False
            },
        ]
    }
    
    
    :returns: 
    CreateClusterSnapshot and  CopyClusterSnapshot returns status as "creating".
    DescribeClusterSnapshots returns status as "creating", "available", "final snapshot", or "failed".
    DeleteClusterSnapshot returns status as "deleted".
    
    """
    pass