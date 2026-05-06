def modify_replication_group(ReplicationGroupId=None, ReplicationGroupDescription=None, PrimaryClusterId=None, SnapshottingClusterId=None, AutomaticFailoverEnabled=None, CacheSecurityGroupNames=None, SecurityGroupIds=None, PreferredMaintenanceWindow=None, NotificationTopicArn=None, CacheParameterGroupName=None, NotificationTopicStatus=None, ApplyImmediately=None, EngineVersion=None, AutoMinorVersionUpgrade=None, SnapshotRetentionLimit=None, SnapshotWindow=None, CacheNodeType=None, NodeGroupId=None):
    """
    Modifies the settings for a replication group.
    See also: AWS API Documentation
    
    
    :example: response = client.modify_replication_group(
        ReplicationGroupId='string',
        ReplicationGroupDescription='string',
        PrimaryClusterId='string',
        SnapshottingClusterId='string',
        AutomaticFailoverEnabled=True|False,
        CacheSecurityGroupNames=[
            'string',
        ],
        SecurityGroupIds=[
            'string',
        ],
        PreferredMaintenanceWindow='string',
        NotificationTopicArn='string',
        CacheParameterGroupName='string',
        NotificationTopicStatus='string',
        ApplyImmediately=True|False,
        EngineVersion='string',
        AutoMinorVersionUpgrade=True|False,
        SnapshotRetentionLimit=123,
        SnapshotWindow='string',
        CacheNodeType='string',
        NodeGroupId='string'
    )
    
    
    :type ReplicationGroupId: string
    :param ReplicationGroupId: [REQUIRED]
            The identifier of the replication group to modify.
            

    :type ReplicationGroupDescription: string
    :param ReplicationGroupDescription: A description for the replication group. Maximum length is 255 characters.

    :type PrimaryClusterId: string
    :param PrimaryClusterId: For replication groups with a single primary, if this parameter is specified, ElastiCache promotes the specified cluster in the specified replication group to the primary role. The nodes of all other clusters in the replication group are read replicas.

    :type SnapshottingClusterId: string
    :param SnapshottingClusterId: The cache cluster ID that is used as the daily snapshot source for the replication group. This parameter cannot be set for Redis (cluster mode enabled) replication groups.

    :type AutomaticFailoverEnabled: boolean
    :param AutomaticFailoverEnabled: Determines whether a read replica is automatically promoted to read/write primary if the existing primary encounters a failure.
            Valid values: true | false
            Note
            ElastiCache Multi-AZ replication groups are not supported on:
            Redis versions earlier than 2.8.6.
            Redis (cluster mode disabled):T1 and T2 cache node types. Redis (cluster mode enabled): T1 node types.
            

    :type CacheSecurityGroupNames: list
    :param CacheSecurityGroupNames: A list of cache security group names to authorize for the clusters in this replication group. This change is asynchronously applied as soon as possible.
            This parameter can be used only with replication group containing cache clusters running outside of an Amazon Virtual Private Cloud (Amazon VPC).
            Constraints: Must contain no more than 255 alphanumeric characters. Must not be Default .
            (string) --
            

    :type SecurityGroupIds: list
    :param SecurityGroupIds: Specifies the VPC Security Groups associated with the cache clusters in the replication group.
            This parameter can be used only with replication group containing cache clusters running in an Amazon Virtual Private Cloud (Amazon VPC).
            (string) --
            

    :type PreferredMaintenanceWindow: string
    :param PreferredMaintenanceWindow: Specifies the weekly time range during which maintenance on the cluster is performed. It is specified as a range in the format ddd:hh24:mi-ddd:hh24:mi (24H Clock UTC). The minimum maintenance window is a 60 minute period.
            Valid values for ddd are:
            sun
            mon
            tue
            wed
            thu
            fri
            sat
            Example: sun:23:00-mon:01:30
            

    :type NotificationTopicArn: string
    :param NotificationTopicArn: The Amazon Resource Name (ARN) of the Amazon SNS topic to which notifications are sent.
            Note
            The Amazon SNS topic owner must be same as the replication group owner.
            

    :type CacheParameterGroupName: string
    :param CacheParameterGroupName: The name of the cache parameter group to apply to all of the clusters in this replication group. This change is asynchronously applied as soon as possible for parameters when the ApplyImmediately parameter is specified as true for this request.

    :type NotificationTopicStatus: string
    :param NotificationTopicStatus: The status of the Amazon SNS notification topic for the replication group. Notifications are sent only if the status is active .
            Valid values: active | inactive
            

    :type ApplyImmediately: boolean
    :param ApplyImmediately: If true , this parameter causes the modifications in this request and any pending modifications to be applied, asynchronously and as soon as possible, regardless of the PreferredMaintenanceWindow setting for the replication group.
            If false , changes to the nodes in the replication group are applied on the next maintenance reboot, or the next failure reboot, whichever occurs first.
            Valid values: true | false
            Default: false
            

    :type EngineVersion: string
    :param EngineVersion: The upgraded version of the cache engine to be run on the cache clusters in the replication group.
            Important: You can upgrade to a newer engine version (see Selecting a Cache Engine and Version ), but you cannot downgrade to an earlier engine version. If you want to use an earlier engine version, you must delete the existing replication group and create it anew with the earlier engine version.
            

    :type AutoMinorVersionUpgrade: boolean
    :param AutoMinorVersionUpgrade: This parameter is currently disabled.

    :type SnapshotRetentionLimit: integer
    :param SnapshotRetentionLimit: The number of days for which ElastiCache retains automatic node group (shard) snapshots before deleting them. For example, if you set SnapshotRetentionLimit to 5, a snapshot that was taken today is retained for 5 days before being deleted.
            Important If the value of SnapshotRetentionLimit is set to zero (0), backups are turned off.
            

    :type SnapshotWindow: string
    :param SnapshotWindow: The daily time range (in UTC) during which ElastiCache begins taking a daily snapshot of the node group (shard) specified by SnapshottingClusterId .
            Example: 05:00-09:00
            If you do not specify this parameter, ElastiCache automatically chooses an appropriate time range.
            

    :type CacheNodeType: string
    :param CacheNodeType: A valid cache node type that you want to scale this replication group to.

    :type NodeGroupId: string
    :param NodeGroupId: The name of the Node Group (called shard in the console).

    :rtype: dict
    :return: {
        'ReplicationGroup': {
            'ReplicationGroupId': 'string',
            'Description': 'string',
            'Status': 'string',
            'PendingModifiedValues': {
                'PrimaryClusterId': 'string',
                'AutomaticFailoverStatus': 'enabled'|'disabled'
            },
            'MemberClusters': [
                'string',
            ],
            'NodeGroups': [
                {
                    'NodeGroupId': 'string',
                    'Status': 'string',
                    'PrimaryEndpoint': {
                        'Address': 'string',
                        'Port': 123
                    },
                    'Slots': 'string',
                    'NodeGroupMembers': [
                        {
                            'CacheClusterId': 'string',
                            'CacheNodeId': 'string',
                            'ReadEndpoint': {
                                'Address': 'string',
                                'Port': 123
                            },
                            'PreferredAvailabilityZone': 'string',
                            'CurrentRole': 'string'
                        },
                    ]
                },
            ],
            'SnapshottingClusterId': 'string',
            'AutomaticFailover': 'enabled'|'disabled'|'enabling'|'disabling',
            'ConfigurationEndpoint': {
                'Address': 'string',
                'Port': 123
            },
            'SnapshotRetentionLimit': 123,
            'SnapshotWindow': 'string',
            'ClusterEnabled': True|False,
            'CacheNodeType': 'string'
        }
    }
    
    
    :returns: 
    Redis versions earlier than 2.8.6.
    Redis (cluster mode disabled):T1 and T2 cache node types. Redis (cluster mode enabled): T1 node types.
    
    """
    pass