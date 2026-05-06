def modify_replication_instance(ReplicationInstanceArn=None, AllocatedStorage=None, ApplyImmediately=None, ReplicationInstanceClass=None, VpcSecurityGroupIds=None, PreferredMaintenanceWindow=None, MultiAZ=None, EngineVersion=None, AllowMajorVersionUpgrade=None, AutoMinorVersionUpgrade=None, ReplicationInstanceIdentifier=None):
    """
    Modifies the replication instance to apply new settings. You can change one or more parameters by specifying these parameters and the new values in the request.
    Some settings are applied during the maintenance window.
    See also: AWS API Documentation
    
    
    :example: response = client.modify_replication_instance(
        ReplicationInstanceArn='string',
        AllocatedStorage=123,
        ApplyImmediately=True|False,
        ReplicationInstanceClass='string',
        VpcSecurityGroupIds=[
            'string',
        ],
        PreferredMaintenanceWindow='string',
        MultiAZ=True|False,
        EngineVersion='string',
        AllowMajorVersionUpgrade=True|False,
        AutoMinorVersionUpgrade=True|False,
        ReplicationInstanceIdentifier='string'
    )
    
    
    :type ReplicationInstanceArn: string
    :param ReplicationInstanceArn: [REQUIRED]
            The Amazon Resource Name (ARN) of the replication instance.
            

    :type AllocatedStorage: integer
    :param AllocatedStorage: The amount of storage (in gigabytes) to be allocated for the replication instance.

    :type ApplyImmediately: boolean
    :param ApplyImmediately: Indicates whether the changes should be applied immediately or during the next maintenance window.

    :type ReplicationInstanceClass: string
    :param ReplicationInstanceClass: The compute and memory capacity of the replication instance.
            Valid Values: dms.t2.micro | dms.t2.small | dms.t2.medium | dms.t2.large | dms.c4.large | dms.c4.xlarge | dms.c4.2xlarge | dms.c4.4xlarge
            

    :type VpcSecurityGroupIds: list
    :param VpcSecurityGroupIds: Specifies the VPC security group to be used with the replication instance. The VPC security group must work with the VPC containing the replication instance.
            (string) --
            

    :type PreferredMaintenanceWindow: string
    :param PreferredMaintenanceWindow: The weekly time range (in UTC) during which system maintenance can occur, which might result in an outage. Changing this parameter does not result in an outage, except in the following situation, and the change is asynchronously applied as soon as possible. If moving this window to the current time, there must be at least 30 minutes between the current time and end of the window to ensure pending changes are applied.
            Default: Uses existing setting
            Format: ddd:hh24:mi-ddd:hh24:mi
            Valid Days: Mon | Tue | Wed | Thu | Fri | Sat | Sun
            Constraints: Must be at least 30 minutes
            

    :type MultiAZ: boolean
    :param MultiAZ: Specifies if the replication instance is a Multi-AZ deployment. You cannot set the AvailabilityZone parameter if the Multi-AZ parameter is set to true .

    :type EngineVersion: string
    :param EngineVersion: The engine version number of the replication instance.

    :type AllowMajorVersionUpgrade: boolean
    :param AllowMajorVersionUpgrade: Indicates that major version upgrades are allowed. Changing this parameter does not result in an outage and the change is asynchronously applied as soon as possible.
            Constraints: This parameter must be set to true when specifying a value for the EngineVersion parameter that is a different major version than the replication instance's current version.
            

    :type AutoMinorVersionUpgrade: boolean
    :param AutoMinorVersionUpgrade: Indicates that minor version upgrades will be applied automatically to the replication instance during the maintenance window. Changing this parameter does not result in an outage except in the following case and the change is asynchronously applied as soon as possible. An outage will result if this parameter is set to true during the maintenance window, and a newer minor version is available, and AWS DMS has enabled auto patching for that engine version.

    :type ReplicationInstanceIdentifier: string
    :param ReplicationInstanceIdentifier: The replication instance identifier. This parameter is stored as a lowercase string.

    :rtype: dict
    :return: {
        'ReplicationInstance': {
            'ReplicationInstanceIdentifier': 'string',
            'ReplicationInstanceClass': 'string',
            'ReplicationInstanceStatus': 'string',
            'AllocatedStorage': 123,
            'InstanceCreateTime': datetime(2015, 1, 1),
            'VpcSecurityGroups': [
                {
                    'VpcSecurityGroupId': 'string',
                    'Status': 'string'
                },
            ],
            'AvailabilityZone': 'string',
            'ReplicationSubnetGroup': {
                'ReplicationSubnetGroupIdentifier': 'string',
                'ReplicationSubnetGroupDescription': 'string',
                'VpcId': 'string',
                'SubnetGroupStatus': 'string',
                'Subnets': [
                    {
                        'SubnetIdentifier': 'string',
                        'SubnetAvailabilityZone': {
                            'Name': 'string'
                        },
                        'SubnetStatus': 'string'
                    },
                ]
            },
            'PreferredMaintenanceWindow': 'string',
            'PendingModifiedValues': {
                'ReplicationInstanceClass': 'string',
                'AllocatedStorage': 123,
                'MultiAZ': True|False,
                'EngineVersion': 'string'
            },
            'MultiAZ': True|False,
            'EngineVersion': 'string',
            'AutoMinorVersionUpgrade': True|False,
            'KmsKeyId': 'string',
            'ReplicationInstanceArn': 'string',
            'ReplicationInstancePublicIpAddress': 'string',
            'ReplicationInstancePrivateIpAddress': 'string',
            'ReplicationInstancePublicIpAddresses': [
                'string',
            ],
            'ReplicationInstancePrivateIpAddresses': [
                'string',
            ],
            'PubliclyAccessible': True|False,
            'SecondaryAvailabilityZone': 'string'
        }
    }
    
    
    :returns: 
    Must contain from 1 to 63 alphanumeric characters or hyphens.
    First character must be a letter.
    Cannot end with a hyphen or contain two consecutive hyphens.
    
    """
    pass