def restore_db_cluster_from_snapshot(AvailabilityZones=None, DBClusterIdentifier=None, SnapshotIdentifier=None, Engine=None, EngineVersion=None, Port=None, DBSubnetGroupName=None, DatabaseName=None, OptionGroupName=None, VpcSecurityGroupIds=None, Tags=None, KmsKeyId=None, EnableIAMDatabaseAuthentication=None):
    """
    Creates a new DB cluster from a DB cluster snapshot. The target DB cluster is created from the source DB cluster restore point with the same configuration as the original source DB cluster, except that the new DB cluster is created with the default security group.
    For more information on Amazon Aurora, see Aurora on Amazon RDS in the Amazon RDS User Guide.
    See also: AWS API Documentation
    
    Examples
    The following example restores an Amazon Aurora DB cluster from a DB cluster snapshot.
    Expected Output:
    
    :example: response = client.restore_db_cluster_from_snapshot(
        AvailabilityZones=[
            'string',
        ],
        DBClusterIdentifier='string',
        SnapshotIdentifier='string',
        Engine='string',
        EngineVersion='string',
        Port=123,
        DBSubnetGroupName='string',
        DatabaseName='string',
        OptionGroupName='string',
        VpcSecurityGroupIds=[
            'string',
        ],
        Tags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        KmsKeyId='string',
        EnableIAMDatabaseAuthentication=True|False
    )
    
    
    :type AvailabilityZones: list
    :param AvailabilityZones: Provides the list of EC2 Availability Zones that instances in the restored DB cluster can be created in.
            (string) --
            

    :type DBClusterIdentifier: string
    :param DBClusterIdentifier: [REQUIRED]
            The name of the DB cluster to create from the DB cluster snapshot. This parameter isn't case-sensitive.
            Constraints:
            Must contain from 1 to 255 alphanumeric characters or hyphens
            First character must be a letter
            Cannot end with a hyphen or contain two consecutive hyphens
            Example: my-snapshot-id
            

    :type SnapshotIdentifier: string
    :param SnapshotIdentifier: [REQUIRED]
            The identifier for the DB cluster snapshot to restore from.
            Constraints:
            Must contain from 1 to 63 alphanumeric characters or hyphens
            First character must be a letter
            Cannot end with a hyphen or contain two consecutive hyphens
            

    :type Engine: string
    :param Engine: [REQUIRED]
            The database engine to use for the new DB cluster.
            Default: The same as source
            Constraint: Must be compatible with the engine of the source
            

    :type EngineVersion: string
    :param EngineVersion: The version of the database engine to use for the new DB cluster.

    :type Port: integer
    :param Port: The port number on which the new DB cluster accepts connections.
            Constraints: Value must be 1150-65535
            Default: The same port as the original DB cluster.
            

    :type DBSubnetGroupName: string
    :param DBSubnetGroupName: The name of the DB subnet group to use for the new DB cluster.
            Constraints: Must contain no more than 255 alphanumeric characters, periods, underscores, spaces, or hyphens. Must not be default.
            Example: mySubnetgroup
            

    :type DatabaseName: string
    :param DatabaseName: The database name for the restored DB cluster.

    :type OptionGroupName: string
    :param OptionGroupName: The name of the option group to use for the restored DB cluster.

    :type VpcSecurityGroupIds: list
    :param VpcSecurityGroupIds: A list of VPC security groups that the new DB cluster will belong to.
            (string) --
            

    :type Tags: list
    :param Tags: The tags to be assigned to the restored DB cluster.
            (dict) --Metadata assigned to an Amazon RDS resource consisting of a key-value pair.
            Key (string) --A key is the required name of the tag. The string value can be from 1 to 128 Unicode characters in length and cannot be prefixed with 'aws:' or 'rds:'. The string can only contain only the set of Unicode letters, digits, white-space, '_', '.', '/', '=', '+', '-' (Java regex: '^([\p{L}\p{Z}\p{N}_.:/=+\-]*)$').
            Value (string) --A value is the optional value of the tag. The string value can be from 1 to 256 Unicode characters in length and cannot be prefixed with 'aws:' or 'rds:'. The string can only contain only the set of Unicode letters, digits, white-space, '_', '.', '/', '=', '+', '-' (Java regex: '^([\p{L}\p{Z}\p{N}_.:/=+\-]*)$').
            
            

    :type KmsKeyId: string
    :param KmsKeyId: The KMS key identifier to use when restoring an encrypted DB cluster from a DB cluster snapshot.
            The KMS key identifier is the Amazon Resource Name (ARN) for the KMS encryption key. If you are restoring a DB cluster with the same AWS account that owns the KMS encryption key used to encrypt the new DB cluster, then you can use the KMS key alias instead of the ARN for the KMS encryption key.
            If you do not specify a value for the KmsKeyId parameter, then the following will occur:
            If the DB cluster snapshot is encrypted, then the restored DB cluster is encrypted using the KMS key that was used to encrypt the DB cluster snapshot.
            If the DB cluster snapshot is not encrypted, then the restored DB cluster is encrypted using the specified encryption key.
            

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