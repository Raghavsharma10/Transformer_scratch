def register_image(DryRun=None, ImageLocation=None, Name=None, Description=None, Architecture=None, KernelId=None, RamdiskId=None, BillingProducts=None, RootDeviceName=None, BlockDeviceMappings=None, VirtualizationType=None, SriovNetSupport=None, EnaSupport=None):
    """
    Registers an AMI. When you're creating an AMI, this is the final step you must complete before you can launch an instance from the AMI. For more information about creating AMIs, see Creating Your Own AMIs in the Amazon Elastic Compute Cloud User Guide .
    You can also use RegisterImage to create an Amazon EBS-backed Linux AMI from a snapshot of a root device volume. You specify the snapshot using the block device mapping. For more information, see Launching a Linux Instance from a Backup in the Amazon Elastic Compute Cloud User Guide .
    You can't register an image where a secondary (non-root) snapshot has AWS Marketplace product codes.
    Some Linux distributions, such as Red Hat Enterprise Linux (RHEL) and SUSE Linux Enterprise Server (SLES), use the EC2 billing product code associated with an AMI to verify the subscription status for package updates. Creating an AMI from an EBS snapshot does not maintain this billing code, and subsequent instances launched from such an AMI will not be able to connect to package update infrastructure. To create an AMI that must retain billing codes, see  CreateImage .
    If needed, you can deregister an AMI at any time. Any modifications you make to an AMI backed by an instance store volume invalidates its registration. If you make changes to an image, deregister the previous image and register the new image.
    See also: AWS API Documentation
    
    
    :example: response = client.register_image(
        DryRun=True|False,
        ImageLocation='string',
        Name='string',
        Description='string',
        Architecture='i386'|'x86_64',
        KernelId='string',
        RamdiskId='string',
        BillingProducts=[
            'string',
        ],
        RootDeviceName='string',
        BlockDeviceMappings=[
            {
                'VirtualName': 'string',
                'DeviceName': 'string',
                'Ebs': {
                    'SnapshotId': 'string',
                    'VolumeSize': 123,
                    'DeleteOnTermination': True|False,
                    'VolumeType': 'standard'|'io1'|'gp2'|'sc1'|'st1',
                    'Iops': 123,
                    'Encrypted': True|False
                },
                'NoDevice': 'string'
            },
        ],
        VirtualizationType='string',
        SriovNetSupport='string',
        EnaSupport=True|False
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type ImageLocation: string
    :param ImageLocation: The full path to your AMI manifest in Amazon S3 storage.

    :type Name: string
    :param Name: [REQUIRED]
            A name for your AMI.
            Constraints: 3-128 alphanumeric characters, parentheses (()), square brackets ([]), spaces ( ), periods (.), slashes (/), dashes (-), single quotes ('), at-signs (@), or underscores(_)
            

    :type Description: string
    :param Description: A description for your AMI.

    :type Architecture: string
    :param Architecture: The architecture of the AMI.
            Default: For Amazon EBS-backed AMIs, i386 . For instance store-backed AMIs, the architecture specified in the manifest file.
            

    :type KernelId: string
    :param KernelId: The ID of the kernel.

    :type RamdiskId: string
    :param RamdiskId: The ID of the RAM disk.

    :type BillingProducts: list
    :param BillingProducts: The billing product codes. Your account must be authorized to specify billing product codes. Otherwise, you can use the AWS Marketplace to bill for the use of an AMI.
            (string) --
            

    :type RootDeviceName: string
    :param RootDeviceName: The name of the root device (for example, /dev/sda1 , or /dev/xvda ).

    :type BlockDeviceMappings: list
    :param BlockDeviceMappings: One or more block device mapping entries.
            (dict) --Describes a block device mapping.
            VirtualName (string) --The virtual device name (ephemeral N). Instance store volumes are numbered starting from 0. An instance type with 2 available instance store volumes can specify mappings for ephemeral0 and ephemeral1 .The number of available instance store volumes depends on the instance type. After you connect to the instance, you must mount the volume.
            Constraints: For M3 instances, you must specify instance store volumes in the block device mapping for the instance. When you launch an M3 instance, we ignore any instance store volumes specified in the block device mapping for the AMI.
            DeviceName (string) --The device name exposed to the instance (for example, /dev/sdh or xvdh ).
            Ebs (dict) --Parameters used to automatically set up EBS volumes when the instance is launched.
            SnapshotId (string) --The ID of the snapshot.
            VolumeSize (integer) --The size of the volume, in GiB.
            Constraints: 1-16384 for General Purpose SSD (gp2 ), 4-16384 for Provisioned IOPS SSD (io1 ), 500-16384 for Throughput Optimized HDD (st1 ), 500-16384 for Cold HDD (sc1 ), and 1-1024 for Magnetic (standard ) volumes. If you specify a snapshot, the volume size must be equal to or larger than the snapshot size.
            Default: If you're creating the volume from a snapshot and don't specify a volume size, the default is the snapshot size.
            DeleteOnTermination (boolean) --Indicates whether the EBS volume is deleted on instance termination.
            VolumeType (string) --The volume type: gp2 , io1 , st1 , sc1 , or standard .
            Default: standard
            Iops (integer) --The number of I/O operations per second (IOPS) that the volume supports. For io1 , this represents the number of IOPS that are provisioned for the volume. For gp2 , this represents the baseline performance of the volume and the rate at which the volume accumulates I/O credits for bursting. For more information about General Purpose SSD baseline performance, I/O credits, and bursting, see Amazon EBS Volume Types in the Amazon Elastic Compute Cloud User Guide .
            Constraint: Range is 100-20000 IOPS for io1 volumes and 100-10000 IOPS for gp2 volumes.
            Condition: This parameter is required for requests to create io1 volumes; it is not used in requests to create gp2 , st1 , sc1 , or standard volumes.
            Encrypted (boolean) --Indicates whether the EBS volume is encrypted. Encrypted Amazon EBS volumes may only be attached to instances that support Amazon EBS encryption.
            NoDevice (string) --Suppresses the specified device included in the block device mapping of the AMI.
            
            

    :type VirtualizationType: string
    :param VirtualizationType: The type of virtualization.
            Default: paravirtual
            

    :type SriovNetSupport: string
    :param SriovNetSupport: Set to simple to enable enhanced networking with the Intel 82599 Virtual Function interface for the AMI and any instances that you launch from the AMI.
            There is no way to disable sriovNetSupport at this time.
            This option is supported only for HVM AMIs. Specifying this option with a PV AMI can make instances launched from the AMI unreachable.
            

    :type EnaSupport: boolean
    :param EnaSupport: Set to true to enable enhanced networking with ENA for the AMI and any instances that you launch from the AMI.
            This option is supported only for HVM AMIs. Specifying this option with a PV AMI can make instances launched from the AMI unreachable.
            

    :rtype: dict
    :return: {
        'ImageId': 'string'
    }
    
    
    """
    pass