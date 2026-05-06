def create_replication_instance(ReplicationInstanceIdentifier=None, AllocatedStorage=None, ReplicationInstanceClass=None, VpcSecurityGroupIds=None, AvailabilityZone=None, ReplicationSubnetGroupIdentifier=None, PreferredMaintenanceWindow=None, MultiAZ=None, EngineVersion=None, AutoMinorVersionUpgrade=None, Tags=None, KmsKeyId=None, PubliclyAccessible=None):
    """
    Creates the replication instance using the specified parameters.
    See also: AWS API Documentation
    
    
    :example: response = client.create_replication_instance(
        ReplicationInstanceIdentifier='string',
        AllocatedStorage=123,
        ReplicationInstanceClass='string',
        VpcSecurityGroupIds=[
            'string',
        ],
        AvailabilityZone='string',
        ReplicationSubnetGroupIdentifier='string',
        PreferredMaintenanceWindow='string',
        MultiAZ=True|False,
        EngineVersion='string',
        AutoMinorVersionUpgrade=True|False,
        Tags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        KmsKeyId='string',
        PubliclyAccessible=True|False
    )
    
    
    :type ReplicationInstanceIdentifier: string
    :param ReplicationInstanceIdentifier: [REQUIRED]
            The replication instance identifier. This parameter is stored as a lowercase string.
            Constraints:
            Must contain from 1 to 63 alphanumeric characters or hyphens.
            First character must be a letter.
            Cannot end with a hyphen or contain two consecutive hyphens.
            Example: myrepinstance
            

    :type AllocatedStorage: integer
    :param AllocatedStorage: The amount of storage (in gigabytes) to be initially allocated for the replication instance.

    :type ReplicationInstanceClass: string
    :param ReplicationInstanceClass: [REQUIRED]
            The compute and memory capacity of the replication instance as specified by the replication instance class.
            Valid Values: dms.t2.micro | dms.t2.small | dms.t2.medium | dms.t2.large | dms.c4.large | dms.c4.xlarge | dms.c4.2xlarge | dms.c4.4xlarge
            

    :type VpcSecurityGroupIds: list
    :param VpcSecurityGroupIds: Specifies the VPC security group to be used with the replication instance. The VPC security group must work with the VPC containing the replication instance.
            (string) --
            

    :type AvailabilityZone: string
    :param AvailabilityZone: The EC2 Availability Zone that the replication instance will be created in.
            Default: A random, system-chosen Availability Zone in the endpoint's region.
            Example: us-east-1d
            

    :type ReplicationSubnetGroupIdentifier: string
    :param ReplicationSubnetGroupIdentifier: A subnet group to associate with the replication instance.

    :type PreferredMaintenanceWindow: string
    :param PreferredMaintenanceWindow: The weekly time range during which system maintenance can occur, in Universal Coordinated Time (UTC).
            Format: ddd:hh24:mi-ddd:hh24:mi
            Default: A 30-minute window selected at random from an 8-hour block of time per region, occurring on a random day of the week.
            Valid Days: Mon, Tue, Wed, Thu, Fri, Sat, Sun
            Constraints: Minimum 30-minute window.
            

    :type MultiAZ: boolean
    :param MultiAZ: Specifies if the replication instance is a Multi-AZ deployment. You cannot set the AvailabilityZone parameter if the Multi-AZ parameter is set to true .

    :type EngineVersion: string
    :param EngineVersion: The engine version number of the replication instance.

    :type AutoMinorVersionUpgrade: boolean
    :param AutoMinorVersionUpgrade: Indicates that minor engine upgrades will be applied automatically to the replication instance during the maintenance window.
            Default: true
            

    :type Tags: list
    :param Tags: Tags to be associated with the replication instance.
            (dict) --
            Key (string) --A key is the required name of the tag. The string value can be from 1 to 128 Unicode characters in length and cannot be prefixed with 'aws:' or 'dms:'. The string can only contain only the set of Unicode letters, digits, white-space, '_', '.', '/', '=', '+', '-' (Java regex: '^([\p{L}\p{Z}\p{N}_.:/=+\-]*)$').
            Value (string) --A value is the optional value of the tag. The string value can be from 1 to 256 Unicode characters in length and cannot be prefixed with 'aws:' or 'dms:'. The string can only contain only the set of Unicode letters, digits, white-space, '_', '.', '/', '=', '+', '-' (Java regex: '^([\p{L}\p{Z}\p{N}_.:/=+\-]*)$').
            
            

    :type KmsKeyId: string
    :param KmsKeyId: The KMS key identifier that will be used to encrypt the content on the replication instance. If you do not specify a value for the KmsKeyId parameter, then AWS DMS will use your default encryption key. AWS KMS creates the default encryption key for your AWS account. Your AWS account has a different default encryption key for each AWS region.

    :type PubliclyAccessible: boolean
    :param PubliclyAccessible: Specifies the accessibility options for the replication instance. A value of true represents an instance with a public IP address. A value of false represents an instance with a private IP address. The default value is true .

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