def create_instance(StackId=None, LayerIds=None, InstanceType=None, AutoScalingType=None, Hostname=None, Os=None, AmiId=None, SshKeyName=None, AvailabilityZone=None, VirtualizationType=None, SubnetId=None, Architecture=None, RootDeviceType=None, BlockDeviceMappings=None, InstallUpdatesOnBoot=None, EbsOptimized=None, AgentVersion=None, Tenancy=None):
    """
    Creates an instance in a specified stack. For more information, see Adding an Instance to a Layer .
    See also: AWS API Documentation
    
    
    :example: response = client.create_instance(
        StackId='string',
        LayerIds=[
            'string',
        ],
        InstanceType='string',
        AutoScalingType='load'|'timer',
        Hostname='string',
        Os='string',
        AmiId='string',
        SshKeyName='string',
        AvailabilityZone='string',
        VirtualizationType='string',
        SubnetId='string',
        Architecture='x86_64'|'i386',
        RootDeviceType='ebs'|'instance-store',
        BlockDeviceMappings=[
            {
                'DeviceName': 'string',
                'NoDevice': 'string',
                'VirtualName': 'string',
                'Ebs': {
                    'SnapshotId': 'string',
                    'Iops': 123,
                    'VolumeSize': 123,
                    'VolumeType': 'gp2'|'io1'|'standard',
                    'DeleteOnTermination': True|False
                }
            },
        ],
        InstallUpdatesOnBoot=True|False,
        EbsOptimized=True|False,
        AgentVersion='string',
        Tenancy='string'
    )
    
    
    :type StackId: string
    :param StackId: [REQUIRED]
            The stack ID.
            

    :type LayerIds: list
    :param LayerIds: [REQUIRED]
            An array that contains the instance's layer IDs.
            (string) --
            

    :type InstanceType: string
    :param InstanceType: [REQUIRED]
            The instance type, such as t2.micro . For a list of supported instance types, open the stack in the console, choose Instances , and choose + Instance . The Size list contains the currently supported types. For more information, see Instance Families and Types . The parameter values that you use to specify the various types are in the API Name column of the Available Instance Types table.
            

    :type AutoScalingType: string
    :param AutoScalingType: For load-based or time-based instances, the type. Windows stacks can use only time-based instances.

    :type Hostname: string
    :param Hostname: The instance host name.

    :type Os: string
    :param Os: The instance's operating system, which must be set to one of the following.
            A supported Linux operating system: An Amazon Linux version, such as Amazon Linux 2016.09 , Amazon Linux 2016.03 , Amazon Linux 2015.09 , or Amazon Linux 2015.03 .
            A supported Ubuntu operating system, such as Ubuntu 16.04 LTS , Ubuntu 14.04 LTS , or Ubuntu 12.04 LTS .
            CentOS Linux 7
            Red Hat Enterprise Linux 7
            A supported Windows operating system, such as Microsoft Windows Server 2012 R2 Base , Microsoft Windows Server 2012 R2 with SQL Server Express , Microsoft Windows Server 2012 R2 with SQL Server Standard , or Microsoft Windows Server 2012 R2 with SQL Server Web .
            A custom AMI: Custom .
            For more information on the supported operating systems, see AWS OpsWorks Stacks Operating Systems .
            The default option is the current Amazon Linux version. If you set this parameter to Custom , you must use the CreateInstance action's AmiId parameter to specify the custom AMI that you want to use. Block device mappings are not supported if the value is Custom . For more information on the supported operating systems, see Operating Systems For more information on how to use custom AMIs with AWS OpsWorks Stacks, see Using Custom AMIs .
            

    :type AmiId: string
    :param AmiId: A custom AMI ID to be used to create the instance. The AMI should be based on one of the supported operating systems. For more information, see Using Custom AMIs .
            Note
            If you specify a custom AMI, you must set Os to Custom .
            

    :type SshKeyName: string
    :param SshKeyName: The instance's Amazon EC2 key-pair name.

    :type AvailabilityZone: string
    :param AvailabilityZone: The instance Availability Zone. For more information, see Regions and Endpoints .

    :type VirtualizationType: string
    :param VirtualizationType: The instance's virtualization type, paravirtual or hvm .

    :type SubnetId: string
    :param SubnetId: The ID of the instance's subnet. If the stack is running in a VPC, you can use this parameter to override the stack's default subnet ID value and direct AWS OpsWorks Stacks to launch the instance in a different subnet.

    :type Architecture: string
    :param Architecture: The instance architecture. The default option is x86_64 . Instance types do not necessarily support both architectures. For a list of the architectures that are supported by the different instance types, see Instance Families and Types .

    :type RootDeviceType: string
    :param RootDeviceType: The instance root device type. For more information, see Storage for the Root Device .

    :type BlockDeviceMappings: list
    :param BlockDeviceMappings: An array of BlockDeviceMapping objects that specify the instance's block devices. For more information, see Block Device Mapping . Note that block device mappings are not supported for custom AMIs.
            (dict) --Describes a block device mapping. This data type maps directly to the Amazon EC2 BlockDeviceMapping data type.
            DeviceName (string) --The device name that is exposed to the instance, such as /dev/sdh . For the root device, you can use the explicit device name or you can set this parameter to ROOT_DEVICE and AWS OpsWorks Stacks will provide the correct device name.
            NoDevice (string) --Suppresses the specified device included in the AMI's block device mapping.
            VirtualName (string) --The virtual device name. For more information, see BlockDeviceMapping .
            Ebs (dict) --An EBSBlockDevice that defines how to configure an Amazon EBS volume when the instance is launched.
            SnapshotId (string) --The snapshot ID.
            Iops (integer) --The number of I/O operations per second (IOPS) that the volume supports. For more information, see EbsBlockDevice .
            VolumeSize (integer) --The volume size, in GiB. For more information, see EbsBlockDevice .
            VolumeType (string) --The volume type. gp2 for General Purpose (SSD) volumes, io1 for Provisioned IOPS (SSD) volumes, and standard for Magnetic volumes.
            DeleteOnTermination (boolean) --Whether the volume is deleted on instance termination.
            
            

    :type InstallUpdatesOnBoot: boolean
    :param InstallUpdatesOnBoot: Whether to install operating system and package updates when the instance boots. The default value is true . To control when updates are installed, set this value to false . You must then update your instances manually by using CreateDeployment to run the update_dependencies stack command or by manually running yum (Amazon Linux) or apt-get (Ubuntu) on the instances.
            Note
            We strongly recommend using the default value of true to ensure that your instances have the latest security updates.
            

    :type EbsOptimized: boolean
    :param EbsOptimized: Whether to create an Amazon EBS-optimized instance.

    :type AgentVersion: string
    :param AgentVersion: The default AWS OpsWorks Stacks agent version. You have the following options:
            INHERIT - Use the stack's default agent version setting.
            version_number - Use the specified agent version. This value overrides the stack's default setting. To update the agent version, edit the instance configuration and specify a new version. AWS OpsWorks Stacks then automatically installs that version on the instance.
            The default setting is INHERIT . To specify an agent version, you must use the complete version number, not the abbreviated number shown on the console. For a list of available agent version numbers, call DescribeAgentVersions . AgentVersion cannot be set to Chef 12.2.
            

    :type Tenancy: string
    :param Tenancy: The instance's tenancy option. The default option is no tenancy, or if the instance is running in a VPC, inherit tenancy settings from the VPC. The following are valid values for this parameter: dedicated , default , or host . Because there are costs associated with changes in tenancy options, we recommend that you research tenancy options before choosing them for your instances. For more information about dedicated hosts, see Dedicated Hosts Overview and Amazon EC2 Dedicated Hosts . For more information about dedicated instances, see Dedicated Instances and Amazon EC2 Dedicated Instances .

    :rtype: dict
    :return: {
        'InstanceId': 'string'
    }
    
    
    """
    pass