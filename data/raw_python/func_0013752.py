def modify_db_cluster(DBClusterIdentifier=None, NewDBClusterIdentifier=None, ApplyImmediately=None, BackupRetentionPeriod=None, DBClusterParameterGroupName=None, VpcSecurityGroupIds=None, Port=None, MasterUserPassword=None, OptionGroupName=None, PreferredBackupWindow=None, PreferredMaintenanceWindow=None, EnableIAMDatabaseAuthentication=None):
    """
    Modify a setting for an Amazon Aurora DB cluster. You can change one or more database configuration parameters by specifying these parameters and the new values in the request. For more information on Amazon Aurora, see Aurora on Amazon RDS in the Amazon RDS User Guide.
    See also: AWS API Documentation
    
    Examples
    This example changes the specified settings for the specified DB cluster.
    Expected Output:
    
    :example: response = client.modify_db_cluster(
        DBClusterIdentifier='string',
        NewDBClusterIdentifier='string',
        ApplyImmediately=True|False,
        BackupRetentionPeriod=123,
        DBClusterParameterGroupName='string',
        VpcSecurityGroupIds=[
            'string',
        ],
        Port=123,
        MasterUserPassword='string',
        OptionGroupName='string',
        PreferredBackupWindow='string',
        PreferredMaintenanceWindow='string',
        EnableIAMDatabaseAuthentication=True|False
    )
    
    
    :type DBClusterIdentifier: string
    :param DBClusterIdentifier: [REQUIRED]
            The DB cluster identifier for the cluster being modified. This parameter is not case-sensitive.
            Constraints:
            Must be the identifier for an existing DB cluster.
            Must contain from 1 to 63 alphanumeric characters or hyphens.
            First character must be a letter.
            Cannot end with a hyphen or contain two consecutive hyphens.
            

    :type NewDBClusterIdentifier: string
    :param NewDBClusterIdentifier: The new DB cluster identifier for the DB cluster when renaming a DB cluster. This value is stored as a lowercase string.
            Constraints:
            Must contain from 1 to 63 alphanumeric characters or hyphens
            First character must be a letter
            Cannot end with a hyphen or contain two consecutive hyphens
            Example: my-cluster2
            

    :type ApplyImmediately: boolean
    :param ApplyImmediately: A value that specifies whether the modifications in this request and any pending modifications are asynchronously applied as soon as possible, regardless of the PreferredMaintenanceWindow setting for the DB cluster. If this parameter is set to false , changes to the DB cluster are applied during the next maintenance window.
            The ApplyImmediately parameter only affects the NewDBClusterIdentifier and MasterUserPassword values. If you set the ApplyImmediately parameter value to false, then changes to the NewDBClusterIdentifier and MasterUserPassword values are applied during the next maintenance window. All other changes are applied immediately, regardless of the value of the ApplyImmediately parameter.
            Default: false
            

    :type BackupRetentionPeriod: integer
    :param BackupRetentionPeriod: The number of days for which automated backups are retained. You must specify a minimum value of 1.
            Default: 1
            Constraints:
            Must be a value from 1 to 35
            

    :type DBClusterParameterGroupName: string
    :param DBClusterParameterGroupName: The name of the DB cluster parameter group to use for the DB cluster.

    :type VpcSecurityGroupIds: list
    :param VpcSecurityGroupIds: A list of VPC security groups that the DB cluster will belong to.
            (string) --
            

    :type Port: integer
    :param Port: The port number on which the DB cluster accepts connections.
            Constraints: Value must be 1150-65535
            Default: The same port as the original DB cluster.
            

    :type MasterUserPassword: string
    :param MasterUserPassword: The new password for the master database user. This password can contain any printable ASCII character except '/', ''', or '@'.
            Constraints: Must contain from 8 to 41 characters.
            

    :type OptionGroupName: string
    :param OptionGroupName: A value that indicates that the DB cluster should be associated with the specified option group. Changing this parameter does not result in an outage except in the following case, and the change is applied during the next maintenance window unless the ApplyImmediately parameter is set to true for this request. If the parameter change results in an option group that enables OEM, this change can cause a brief (sub-second) period during which new connections are rejected but existing connections are not interrupted.
            Permanent options cannot be removed from an option group. The option group cannot be removed from a DB cluster once it is associated with a DB cluster.
            

    :type PreferredBackupWindow: string
    :param PreferredBackupWindow: The daily time range during which automated backups are created if automated backups are enabled, using the BackupRetentionPeriod parameter.
            Default: A 30-minute window selected at random from an 8-hour block of time per region. To see the time blocks available, see Adjusting the Preferred Maintenance Window in the Amazon RDS User Guide.
            Constraints:
            Must be in the format hh24:mi-hh24:mi .
            Times should be in Universal Coordinated Time (UTC).
            Must not conflict with the preferred maintenance window.
            Must be at least 30 minutes.
            

    :type PreferredMaintenanceWindow: string
    :param PreferredMaintenanceWindow: The weekly time range during which system maintenance can occur, in Universal Coordinated Time (UTC).
            Format: ddd:hh24:mi-ddd:hh24:mi
            Default: A 30-minute window selected at random from an 8-hour block of time per region, occurring on a random day of the week. To see the time blocks available, see Adjusting the Preferred Maintenance Window in the Amazon RDS User Guide.
            Valid Days: Mon, Tue, Wed, Thu, Fri, Sat, Sun
            Constraints: Minimum 30-minute window.
            

    :type EnableIAMDatabaseAuthentication: boolean
    :param EnableIAMDatabaseAuthentication: A Boolean value that is true to enable mapping of AWS Identity and Access Management (IAM) accounts to database accounts, and otherwise false.
            Default: false
            

    :rtype: dict
    :return: {
        'DBCluster': {
            'AllocatedStorage': 123,
            'AvailabilityZones': [
                'string',
            ],
            'BackupRetentionPeriod': 123,
            'CharacterSetName': 'string',
            'DatabaseName': 'string',
            'DBClusterIdentifier': 'string',
            'DBClusterParameterGroup': 'string',
            'DBSubnetGroup': 'string',
            'Status': 'string',
            'PercentProgress': 'string',
            'EarliestRestorableTime': datetime(2015, 1, 1),
            'Endpoint': 'string',
            'ReaderEndpoint': 'string',
            'MultiAZ': True|False,
            'Engine': 'string',
            'EngineVersion': 'string',
            'LatestRestorableTime': datetime(2015, 1, 1),
            'Port': 123,
            'MasterUsername': 'string',
            'DBClusterOptionGroupMemberships': [
                {
                    'DBClusterOptionGroupName': 'string',
                    'Status': 'string'
                },
            ],
            'PreferredBackupWindow': 'string',
            'PreferredMaintenanceWindow': 'string',
            'ReplicationSourceIdentifier': 'string',
            'ReadReplicaIdentifiers': [
                'string',
            ],
            'DBClusterMembers': [
                {
                    'DBInstanceIdentifier': 'string',
                    'IsClusterWriter': True|False,
                    'DBClusterParameterGroupStatus': 'string',
                    'PromotionTier': 123
                },
            ],
            'VpcSecurityGroups': [
                {
                    'VpcSecurityGroupId': 'string',
                    'Status': 'string'
                },
            ],
            'HostedZoneId': 'string',
            'StorageEncrypted': True|False,
            'KmsKeyId': 'string',
            'DbClusterResourceId': 'string',
            'DBClusterArn': 'string',
            'AssociatedRoles': [
                {
                    'RoleArn': 'string',
                    'Status': 'string'
                },
            ],
            'IAMDatabaseAuthenticationEnabled': True|False,
            'ClusterCreateTime': datetime(2015, 1, 1)
        }
    }
    
    
    :returns: 
    CreateDBCluster
    DeleteDBCluster
    FailoverDBCluster
    ModifyDBCluster
    RestoreDBClusterFromSnapshot
    RestoreDBClusterToPointInTime
    
    """
    pass