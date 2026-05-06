def create_volume(DryRun=None, Size=None, SnapshotId=None, AvailabilityZone=None, VolumeType=None, Iops=None, Encrypted=None, KmsKeyId=None, TagSpecifications=None):
    """
    Creates an EBS volume that can be attached to an instance in the same Availability Zone. The volume is created in the regional endpoint that you send the HTTP request to. For more information see Regions and Endpoints .
    You can create a new empty volume or restore a volume from an EBS snapshot. Any AWS Marketplace product codes from the snapshot are propagated to the volume.
    You can create encrypted volumes with the Encrypted parameter. Encrypted volumes may only be attached to instances that support Amazon EBS encryption. Volumes that are created from encrypted snapshots are also automatically encrypted. For more information, see Amazon EBS Encryption in the Amazon Elastic Compute Cloud User Guide .
    You can tag your volumes during creation. For more information, see Tagging Your Amazon EC2 Resources .
    For more information, see Creating an Amazon EBS Volume in the Amazon Elastic Compute Cloud User Guide .
    See also: AWS API Documentation
    
    Examples
    This example creates an 80 GiB General Purpose (SSD) volume in the Availability Zone us-east-1a.
    Expected Output:
    This example creates a new Provisioned IOPS (SSD) volume with 1000 provisioned IOPS from a snapshot in the Availability Zone us-east-1a.
    Expected Output:
    
    :example: response = client.create_volume(
        DryRun=True|False,
        Size=123,
        SnapshotId='string',
        AvailabilityZone='string',
        VolumeType='standard'|'io1'|'gp2'|'sc1'|'st1',
        Iops=123,
        Encrypted=True|False,
        KmsKeyId='string',
        TagSpecifications=[
            {
                'ResourceType': 'customer-gateway'|'dhcp-options'|'image'|'instance'|'internet-gateway'|'network-acl'|'network-interface'|'reserved-instances'|'route-table'|'snapshot'|'spot-instances-request'|'subnet'|'security-group'|'volume'|'vpc'|'vpn-connection'|'vpn-gateway',
                'Tags': [
                    {
                        'Key': 'string',
                        'Value': 'string'
                    },
                ]
            },
        ]
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type Size: integer
    :param Size: The size of the volume, in GiBs.
            Constraints: 1-16384 for gp2 , 4-16384 for io1 , 500-16384 for st1 , 500-16384 for sc1 , and 1-1024 for standard . If you specify a snapshot, the volume size must be equal to or larger than the snapshot size.
            Default: If you're creating the volume from a snapshot and don't specify a volume size, the default is the snapshot size.
            

    :type SnapshotId: string
    :param SnapshotId: The snapshot from which to create the volume.

    :type AvailabilityZone: string
    :param AvailabilityZone: [REQUIRED]
            The Availability Zone in which to create the volume. Use DescribeAvailabilityZones to list the Availability Zones that are currently available to you.
            

    :type VolumeType: string
    :param VolumeType: The volume type. This can be gp2 for General Purpose SSD, io1 for Provisioned IOPS SSD, st1 for Throughput Optimized HDD, sc1 for Cold HDD, or standard for Magnetic volumes.
            Default: standard
            

    :type Iops: integer
    :param Iops: Only valid for Provisioned IOPS SSD volumes. The number of I/O operations per second (IOPS) to provision for the volume, with a maximum ratio of 50 IOPS/GiB.
            Constraint: Range is 100 to 20000 for Provisioned IOPS SSD volumes
            

    :type Encrypted: boolean
    :param Encrypted: Specifies whether the volume should be encrypted. Encrypted Amazon EBS volumes may only be attached to instances that support Amazon EBS encryption. Volumes that are created from encrypted snapshots are automatically encrypted. There is no way to create an encrypted volume from an unencrypted snapshot or vice versa. If your AMI uses encrypted volumes, you can only launch it on supported instance types. For more information, see Amazon EBS Encryption in the Amazon Elastic Compute Cloud User Guide .

    :type KmsKeyId: string
    :param KmsKeyId: The full ARN of the AWS Key Management Service (AWS KMS) customer master key (CMK) to use when creating the encrypted volume. This parameter is only required if you want to use a non-default CMK; if this parameter is not specified, the default CMK for EBS is used. The ARN contains the arn:aws:kms namespace, followed by the region of the CMK, the AWS account ID of the CMK owner, the key namespace, and then the CMK ID. For example, arn:aws:kms:us-east-1 :012345678910 :key/abcd1234-a123-456a-a12b-a123b4cd56ef . If a KmsKeyId is specified, the Encrypted flag must also be set.

    :type TagSpecifications: list
    :param TagSpecifications: The tags to apply to the volume during creation.
            (dict) --The tags to apply to a resource when the resource is being created.
            ResourceType (string) --The type of resource to tag. Currently, the resource types that support tagging on creation are instance and volume .
            Tags (list) --The tags to apply to the resource.
            (dict) --Describes a tag.
            Key (string) --The key of the tag.
            Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws:
            Value (string) --The value of the tag.
            Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.
            
            
            

    :rtype: dict
    :return: {
        'VolumeId': 'string',
        'Size': 123,
        'SnapshotId': 'string',
        'AvailabilityZone': 'string',
        'State': 'creating'|'available'|'in-use'|'deleting'|'deleted'|'error',
        'CreateTime': datetime(2015, 1, 1),
        'Attachments': [
            {
                'VolumeId': 'string',
                'InstanceId': 'string',
                'Device': 'string',
                'State': 'attaching'|'attached'|'detaching'|'detached',
                'AttachTime': datetime(2015, 1, 1),
                'DeleteOnTermination': True|False
            },
        ],
        'Tags': [
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        'VolumeType': 'standard'|'io1'|'gp2'|'sc1'|'st1',
        'Iops': 123,
        'Encrypted': True|False,
        'KmsKeyId': 'string'
    }
    
    
    """
    pass