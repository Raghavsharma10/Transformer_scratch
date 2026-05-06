def create_launch_configuration(LaunchConfigurationName=None, ImageId=None, KeyName=None, SecurityGroups=None, ClassicLinkVPCId=None, ClassicLinkVPCSecurityGroups=None, UserData=None, InstanceId=None, InstanceType=None, KernelId=None, RamdiskId=None, BlockDeviceMappings=None, InstanceMonitoring=None, SpotPrice=None, IamInstanceProfile=None, EbsOptimized=None, AssociatePublicIpAddress=None, PlacementTenancy=None):
    """
    Creates a launch configuration.
    If you exceed your maximum limit of launch configurations, which by default is 100 per region, the call fails. For information about viewing and updating this limit, see  DescribeAccountLimits .
    For more information, see Launch Configurations in the Auto Scaling User Guide .
    See also: AWS API Documentation
    
    Examples
    This example creates a launch configuration.
    Expected Output:
    
    :example: response = client.create_launch_configuration(
        LaunchConfigurationName='string',
        ImageId='string',
        KeyName='string',
        SecurityGroups=[
            'string',
        ],
        ClassicLinkVPCId='string',
        ClassicLinkVPCSecurityGroups=[
            'string',
        ],
        UserData='string',
        InstanceId='string',
        InstanceType='string',
        KernelId='string',
        RamdiskId='string',
        BlockDeviceMappings=[
            {
                'VirtualName': 'string',
                'DeviceName': 'string',
                'Ebs': {
                    'SnapshotId': 'string',
                    'VolumeSize': 123,
                    'VolumeType': 'string',
                    'DeleteOnTermination': True|False,
                    'Iops': 123,
                    'Encrypted': True|False
                },
                'NoDevice': True|False
            },
        ],
        InstanceMonitoring={
            'Enabled': True|False
        },
        SpotPrice='string',
        IamInstanceProfile='string',
        EbsOptimized=True|False,
        AssociatePublicIpAddress=True|False,
        PlacementTenancy='string'
    )
    
    
    :type LaunchConfigurationName: string
    :param LaunchConfigurationName: [REQUIRED]
            The name of the launch configuration. This name must be unique within the scope of your AWS account.
            

    :type ImageId: string
    :param ImageId: The ID of the Amazon Machine Image (AMI) to use to launch your EC2 instances.
            If you do not specify InstanceId , you must specify ImageId .
            For more information, see Finding an AMI in the Amazon Elastic Compute Cloud User Guide .
            

    :type KeyName: string
    :param KeyName: The name of the key pair. For more information, see Amazon EC2 Key Pairs in the Amazon Elastic Compute Cloud User Guide .

    :type SecurityGroups: list
    :param SecurityGroups: One or more security groups with which to associate the instances.
            If your instances are launched in EC2-Classic, you can either specify security group names or the security group IDs. For more information about security groups for EC2-Classic, see Amazon EC2 Security Groups in the Amazon Elastic Compute Cloud User Guide .
            If your instances are launched into a VPC, specify security group IDs. For more information, see Security Groups for Your VPC in the Amazon Virtual Private Cloud User Guide .
            (string) --
            

    :type ClassicLinkVPCId: string
    :param ClassicLinkVPCId: The ID of a ClassicLink-enabled VPC to link your EC2-Classic instances to. This parameter is supported only if you are launching EC2-Classic instances. For more information, see ClassicLink in the Amazon Elastic Compute Cloud User Guide .

    :type ClassicLinkVPCSecurityGroups: list
    :param ClassicLinkVPCSecurityGroups: The IDs of one or more security groups for the specified ClassicLink-enabled VPC. This parameter is required if you specify a ClassicLink-enabled VPC, and is not supported otherwise. For more information, see ClassicLink in the Amazon Elastic Compute Cloud User Guide .
            (string) --
            

    :type UserData: string
    :param UserData: The user data to make available to the launched EC2 instances. For more information, see Instance Metadata and User Data in the Amazon Elastic Compute Cloud User Guide .
            This value will be base64 encoded automatically. Do not base64 encode this value prior to performing the operation.
            

    :type InstanceId: string
    :param InstanceId: The ID of the instance to use to create the launch configuration. The new launch configuration derives attributes from the instance, with the exception of the block device mapping.
            If you do not specify InstanceId , you must specify both ImageId and InstanceType .
            To create a launch configuration with a block device mapping or override any other instance attributes, specify them as part of the same request.
            For more information, see Create a Launch Configuration Using an EC2 Instance in the Auto Scaling User Guide .
            

    :type InstanceType: string
    :param InstanceType: The instance type of the EC2 instance.
            If you do not specify InstanceId , you must specify InstanceType .
            For information about available instance types, see Available Instance Types in the Amazon Elastic Compute Cloud User Guide.
            

    :type KernelId: string
    :param KernelId: The ID of the kernel associated with the AMI.

    :type RamdiskId: string
    :param RamdiskId: The ID of the RAM disk associated with the AMI.

    :type BlockDeviceMappings: list
    :param BlockDeviceMappings: One or more mappings that specify how block devices are exposed to the instance. For more information, see Block Device Mapping in the Amazon Elastic Compute Cloud User Guide .
            (dict) --Describes a block device mapping.
            VirtualName (string) --The name of the virtual device (for example, ephemeral0 ).
            DeviceName (string) -- [REQUIRED]The device name exposed to the EC2 instance (for example, /dev/sdh or xvdh ).
            Ebs (dict) --The information about the Amazon EBS volume.
            SnapshotId (string) --The ID of the snapshot.
            VolumeSize (integer) --The volume size, in GiB. For standard volumes, specify a value from 1 to 1,024. For io1 volumes, specify a value from 4 to 16,384. For gp2 volumes, specify a value from 1 to 16,384. If you specify a snapshot, the volume size must be equal to or larger than the snapshot size.
            Default: If you create a volume from a snapshot and you don't specify a volume size, the default is the snapshot size.
            VolumeType (string) --The volume type. For more information, see Amazon EBS Volume Types in the Amazon Elastic Compute Cloud User Guide .
            Valid values: standard | io1 | gp2
            Default: standard
            DeleteOnTermination (boolean) --Indicates whether the volume is deleted on instance termination.
            Default: true
            Iops (integer) --The number of I/O operations per second (IOPS) to provision for the volume.
            Constraint: Required when the volume type is io1 .
            Encrypted (boolean) --Indicates whether the volume should be encrypted. Encrypted EBS volumes must be attached to instances that support Amazon EBS encryption. Volumes that are created from encrypted snapshots are automatically encrypted. There is no way to create an encrypted volume from an unencrypted snapshot or an unencrypted volume from an encrypted snapshot. For more information, see Amazon EBS Encryption in the Amazon Elastic Compute Cloud User Guide .
            NoDevice (boolean) --Suppresses a device mapping.
            If this parameter is true for the root device, the instance might fail the EC2 health check. Auto Scaling launches a replacement instance if the instance fails the health check.
            
            

    :type InstanceMonitoring: dict
    :param InstanceMonitoring: Enables detailed monitoring (true ) or basic monitoring (false ) for the Auto Scaling instances. The default is true .
            Enabled (boolean) --If true , detailed monitoring is enabled. Otherwise, basic monitoring is enabled.
            

    :type SpotPrice: string
    :param SpotPrice: The maximum hourly price to be paid for any Spot Instance launched to fulfill the request. Spot Instances are launched when the price you specify exceeds the current Spot market price. For more information, see Launching Spot Instances in Your Auto Scaling Group in the Auto Scaling User Guide .

    :type IamInstanceProfile: string
    :param IamInstanceProfile: The name or the Amazon Resource Name (ARN) of the instance profile associated with the IAM role for the instance.
            EC2 instances launched with an IAM role will automatically have AWS security credentials available. You can use IAM roles with Auto Scaling to automatically enable applications running on your EC2 instances to securely access other AWS resources. For more information, see Launch Auto Scaling Instances with an IAM Role in the Auto Scaling User Guide .
            

    :type EbsOptimized: boolean
    :param EbsOptimized: Indicates whether the instance is optimized for Amazon EBS I/O. By default, the instance is not optimized for EBS I/O. The optimization provides dedicated throughput to Amazon EBS and an optimized configuration stack to provide optimal I/O performance. This optimization is not available with all instance types. Additional usage charges apply. For more information, see Amazon EBS-Optimized Instances in the Amazon Elastic Compute Cloud User Guide .

    :type AssociatePublicIpAddress: boolean
    :param AssociatePublicIpAddress: Used for groups that launch instances into a virtual private cloud (VPC). Specifies whether to assign a public IP address to each instance. For more information, see Launching Auto Scaling Instances in a VPC in the Auto Scaling User Guide .
            If you specify this parameter, be sure to specify at least one subnet when you create your group.
            Default: If the instance is launched into a default subnet, the default is true . If the instance is launched into a nondefault subnet, the default is false . For more information, see Supported Platforms in the Amazon Elastic Compute Cloud User Guide .
            

    :type PlacementTenancy: string
    :param PlacementTenancy: The tenancy of the instance. An instance with a tenancy of dedicated runs on single-tenant hardware and can only be launched into a VPC.
            You must set the value of this parameter to dedicated if want to launch Dedicated Instances into a shared tenancy VPC (VPC with instance placement tenancy attribute set to default ).
            If you specify this parameter, be sure to specify at least one subnet when you create your group.
            For more information, see Launching Auto Scaling Instances in a VPC in the Auto Scaling User Guide .
            Valid values: default | dedicated
            

    :return: response = client.create_launch_configuration(
        IamInstanceProfile='my-iam-role',
        ImageId='ami-12345678',
        InstanceType='m3.medium',
        LaunchConfigurationName='my-launch-config',
        SecurityGroups=[
            'sg-eb2af88e',
        ],
    )
    
    print(response)
    
    
    """
    pass