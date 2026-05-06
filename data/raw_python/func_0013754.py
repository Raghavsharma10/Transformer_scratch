def restore_db_cluster_to_point_in_time(DBClusterIdentifier=None, SourceDBClusterIdentifier=None, RestoreToTime=None, UseLatestRestorableTime=None, Port=None, DBSubnetGroupName=None, OptionGroupName=None, VpcSecurityGroupIds=None, Tags=None, KmsKeyId=None, EnableIAMDatabaseAuthentication=None):
    """
    Restores a DB cluster to an arbitrary point in time. Users can restore to any point in time before LatestRestorableTime for up to BackupRetentionPeriod days. The target DB cluster is created from the source DB cluster with the same configuration as the original DB cluster, except that the new DB cluster is created with the default DB security group.
    For more information on Amazon Aurora, see Aurora on Amazon RDS in the Amazon RDS User Guide.
    See also: AWS API Documentation
    
    Examples
    The following example restores a DB cluster to a new DB cluster at a point in time from the source DB cluster.
    Expected Output:
    
    :example: response = client.restore_db_cluster_to_point_in_time(
        DBClusterIdentifier='string',
        SourceDBClusterIdentifier='string',
        RestoreToTime=datetime(2015, 1, 1),
        UseLatestRestorableTime=True|False,
        Port=123,
        DBSubnetGroupName='string',
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
    
    
    :type DBClusterIdentifier: string
    :param DBClusterIdentifier: [REQUIRED]
            The name of the new DB cluster to be created.
            Constraints:
            Must contain from 1 to 63 alphanumeric characters or hyphens
            First character must be a letter
            Cannot end with a hyphen or contain two consecutive hyphens
            

    :type SourceDBClusterIdentifier: string
    :param SourceDBClusterIdentifier: [REQUIRED]
            The identifier of the source DB cluster from which to restore.
            Constraints:
            Must be the identifier of an existing database instance
            Must contain from 1 to 63 alphanumeric characters or hyphens
            First character must be a letter
            Cannot end with a hyphen or contain two consecutive hyphens
            

    :type RestoreToTime: datetime
    :param RestoreToTime: The date and time to restore the DB cluster to.
            Valid Values: Value must be a time in Universal Coordinated Time (UTC) format
            Constraints:
            Must be before the latest restorable time for the DB instance
            Cannot be specified if UseLatestRestorableTime parameter is true
            Example: 2015-03-07T23:45:00Z
            

    :type UseLatestRestorableTime: boolean
    :param UseLatestRestorableTime: A value that is set to true to restore the DB cluster to the latest restorable backup time, and false otherwise.
            Default: false
            Constraints: Cannot be specified if RestoreToTime parameter is provided.
            

    :type Port: integer
    :param Port: The port number on which the new DB cluster accepts connections.
            Constraints: Value must be 1150-65535
            Default: The same port as the original DB cluster.
            

    :type DBSubnetGroupName: string
    :param DBSubnetGroupName: The DB subnet group name to use for the new DB cluster.
            Constraints: Must contain no more than 255 alphanumeric characters, periods, underscores, spaces, or hyphens. Must not be default.
            Example: mySubnetgroup
            

    :type OptionGroupName: string
    :param OptionGroupName: The name of the option group for the new DB cluster.

    :type VpcSecurityGroupIds: list
    :param VpcSecurityGroupIds: A lst of VPC security groups that the new DB cluster belongs to.
            (string) --
            

    :type Tags: list
    :param Tags: A list of tags.
            (dict) --Metadata assigned to an Amazon RDS resource consisting of a key-value pair.
            Key (string) --A key is the required name of the tag. The string value can be from 1 to 128 Unicode characters in length and cannot be prefixed with 'aws:' or 'rds:'. The string can only contain only the set of Unicode letters, digits, white-space, '_', '.', '/', '=', '+', '-' (Java regex: '^([\p{L}\p{Z}\p{N}_.:/=+\-]*)$').
            Value (string) --A value is the optional value of the tag. The string value can be from 1 to 256 Unicode characters in length and cannot be prefixed with 'aws:' or 'rds:'. The string can only contain only the set of Unicode letters, digits, white-space, '_', '.', '/', '=', '+', '-' (Java regex: '^([\p{L}\p{Z}\p{N}_.:/=+\-]*)$').
            
            

    :type KmsKeyId: string
    :param KmsKeyId: The KMS key identifier to use when restoring an encrypted DB cluster from an encrypted DB cluster.
            The KMS key identifier is the Amazon Resource Name (ARN) for the KMS encryption key. If you are restoring a DB cluster with the same AWS account that owns the KMS encryption key used to encrypt the new DB cluster, then you can use the KMS key alias instead of the ARN for the KMS encryption key.
            You can restore to a new DB cluster and encrypt the new DB cluster with a KMS key that is different than the KMS key used to encrypt the source DB cluster. The new DB cluster will be encrypted with the KMS key identified by the KmsKeyId parameter.
            If you do not specify a value for the KmsKeyId parameter, then the following will occur:
            If the DB cluster is encrypted, then the restored DB cluster is encrypted using the KMS key that was used to encrypt the source DB cluster.
            If the DB cluster is not encrypted, then the restored DB cluster is not encrypted.
            If DBClusterIdentifier refers to a DB cluster that is note encrypted, then the restore request is rejected.
            

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