def modify_instance_attribute(DryRun=None, InstanceId=None, Attribute=None, Value=None, BlockDeviceMappings=None, SourceDestCheck=None, DisableApiTermination=None, InstanceType=None, Kernel=None, Ramdisk=None, UserData=None, InstanceInitiatedShutdownBehavior=None, Groups=None, EbsOptimized=None, SriovNetSupport=None, EnaSupport=None):
    """
    Modifies the specified attribute of the specified instance. You can specify only one attribute at a time.
    To modify some attributes, the instance must be stopped. For more information, see Modifying Attributes of a Stopped Instance in the Amazon Elastic Compute Cloud User Guide .
    See also: AWS API Documentation
    
    
    :example: response = client.modify_instance_attribute(
        DryRun=True|False,
        InstanceId='string',
        Attribute='instanceType'|'kernel'|'ramdisk'|'userData'|'disableApiTermination'|'instanceInitiatedShutdownBehavior'|'rootDeviceName'|'blockDeviceMapping'|'productCodes'|'sourceDestCheck'|'groupSet'|'ebsOptimized'|'sriovNetSupport'|'enaSupport',
        Value='string',
        BlockDeviceMappings=[
            {
                'DeviceName': 'string',
                'Ebs': {
                    'VolumeId': 'string',
                    'DeleteOnTermination': True|False
                },
                'VirtualName': 'string',
                'NoDevice': 'string'
            },
        ],
        SourceDestCheck={
            'Value': True|False
        },
        DisableApiTermination={
            'Value': True|False
        },
        InstanceType={
            'Value': 'string'
        },
        Kernel={
            'Value': 'string'
        },
        Ramdisk={
            'Value': 'string'
        },
        UserData={
            'Value': b'bytes'
        },
        InstanceInitiatedShutdownBehavior={
            'Value': 'string'
        },
        Groups=[
            'string',
        ],
        EbsOptimized={
            'Value': True|False
        },
        SriovNetSupport={
            'Value': 'string'
        },
        EnaSupport={
            'Value': True|False
        }
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type InstanceId: string
    :param InstanceId: [REQUIRED]
            The ID of the instance.
            

    :type Attribute: string
    :param Attribute: The name of the attribute.

    :type Value: string
    :param Value: A new value for the attribute. Use only with the kernel , ramdisk , userData , disableApiTermination , or instanceInitiatedShutdownBehavior attribute.

    :type BlockDeviceMappings: list
    :param BlockDeviceMappings: Modifies the DeleteOnTermination attribute for volumes that are currently attached. The volume must be owned by the caller. If no value is specified for DeleteOnTermination , the default is true and the volume is deleted when the instance is terminated.
            To add instance store volumes to an Amazon EBS-backed instance, you must add them when you launch the instance. For more information, see Updating the Block Device Mapping when Launching an Instance in the Amazon Elastic Compute Cloud User Guide .
            (dict) --Describes a block device mapping entry.
            DeviceName (string) --The device name exposed to the instance (for example, /dev/sdh or xvdh ).
            Ebs (dict) --Parameters used to automatically set up EBS volumes when the instance is launched.
            VolumeId (string) --The ID of the EBS volume.
            DeleteOnTermination (boolean) --Indicates whether the volume is deleted on instance termination.
            VirtualName (string) --The virtual device name.
            NoDevice (string) --suppress the specified device included in the block device mapping.
            
            

    :type SourceDestCheck: dict
    :param SourceDestCheck: Specifies whether source/destination checking is enabled. A value of true means that checking is enabled, and false means checking is disabled. This value must be false for a NAT instance to perform NAT.
            Value (boolean) --The attribute value. The valid values are true or false .
            

    :type DisableApiTermination: dict
    :param DisableApiTermination: If the value is true , you can't terminate the instance using the Amazon EC2 console, CLI, or API; otherwise, you can. You cannot use this paramater for Spot Instances.
            Value (boolean) --The attribute value. The valid values are true or false .
            

    :type InstanceType: dict
    :param InstanceType: Changes the instance type to the specified value. For more information, see Instance Types . If the instance type is not valid, the error returned is InvalidInstanceAttributeValue .
            Value (string) --The attribute value. Note that the value is case-sensitive.
            

    :type Kernel: dict
    :param Kernel: Changes the instance's kernel to the specified value. We recommend that you use PV-GRUB instead of kernels and RAM disks. For more information, see PV-GRUB .
            Value (string) --The attribute value. Note that the value is case-sensitive.
            

    :type Ramdisk: dict
    :param Ramdisk: Changes the instance's RAM disk to the specified value. We recommend that you use PV-GRUB instead of kernels and RAM disks. For more information, see PV-GRUB .
            Value (string) --The attribute value. Note that the value is case-sensitive.
            

    :type UserData: dict
    :param UserData: Changes the instance's user data to the specified value. If you are using an AWS SDK or command line tool, Base64-encoding is performed for you, and you can load the text from a file. Otherwise, you must provide Base64-encoded text.
            Value (bytes) --
            

    :type InstanceInitiatedShutdownBehavior: dict
    :param InstanceInitiatedShutdownBehavior: Specifies whether an instance stops or terminates when you initiate shutdown from the instance (using the operating system command for system shutdown).
            Value (string) --The attribute value. Note that the value is case-sensitive.
            

    :type Groups: list
    :param Groups: [EC2-VPC] Changes the security groups of the instance. You must specify at least one security group, even if it's just the default security group for the VPC. You must specify the security group ID, not the security group name.
            (string) --
            

    :type EbsOptimized: dict
    :param EbsOptimized: Specifies whether the instance is optimized for EBS I/O. This optimization provides dedicated throughput to Amazon EBS and an optimized configuration stack to provide optimal EBS I/O performance. This optimization isn't available with all instance types. Additional usage charges apply when using an EBS Optimized instance.
            Value (boolean) --The attribute value. The valid values are true or false .
            

    :type SriovNetSupport: dict
    :param SriovNetSupport: Set to simple to enable enhanced networking with the Intel 82599 Virtual Function interface for the instance.
            There is no way to disable enhanced networking with the Intel 82599 Virtual Function interface at this time.
            This option is supported only for HVM instances. Specifying this option with a PV instance can make it unreachable.
            Value (string) --The attribute value. Note that the value is case-sensitive.
            

    :type EnaSupport: dict
    :param EnaSupport: Set to true to enable enhanced networking with ENA for the instance.
            This option is supported only for HVM instances. Specifying this option with a PV instance can make it unreachable.
            Value (boolean) --The attribute value. The valid values are true or false .
            

    """
    pass