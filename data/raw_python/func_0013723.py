def import_image(DryRun=None, Description=None, DiskContainers=None, LicenseType=None, Hypervisor=None, Architecture=None, Platform=None, ClientData=None, ClientToken=None, RoleName=None):
    """
    Import single or multi-volume disk images or EBS snapshots into an Amazon Machine Image (AMI). For more information, see Importing a VM as an Image Using VM Import/Export in the VM Import/Export User Guide .
    See also: AWS API Documentation
    
    
    :example: response = client.import_image(
        DryRun=True|False,
        Description='string',
        DiskContainers=[
            {
                'Description': 'string',
                'Format': 'string',
                'Url': 'string',
                'UserBucket': {
                    'S3Bucket': 'string',
                    'S3Key': 'string'
                },
                'DeviceName': 'string',
                'SnapshotId': 'string'
            },
        ],
        LicenseType='string',
        Hypervisor='string',
        Architecture='string',
        Platform='string',
        ClientData={
            'UploadStart': datetime(2015, 1, 1),
            'UploadEnd': datetime(2015, 1, 1),
            'UploadSize': 123.0,
            'Comment': 'string'
        },
        ClientToken='string',
        RoleName='string'
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type Description: string
    :param Description: A description string for the import image task.

    :type DiskContainers: list
    :param DiskContainers: Information about the disk containers.
            (dict) --Describes the disk container object for an import image task.
            Description (string) --The description of the disk image.
            Format (string) --The format of the disk image being imported.
            Valid values: RAW | VHD | VMDK | OVA
            Url (string) --The URL to the Amazon S3-based disk image being imported. The URL can either be a https URL (https://..) or an Amazon S3 URL (s3://..)
            UserBucket (dict) --The S3 bucket for the disk image.
            S3Bucket (string) --The name of the S3 bucket where the disk image is located.
            S3Key (string) --The file name of the disk image.
            DeviceName (string) --The block device mapping for the disk.
            SnapshotId (string) --The ID of the EBS snapshot to be used for importing the snapshot.
            
            

    :type LicenseType: string
    :param LicenseType: The license type to be used for the Amazon Machine Image (AMI) after importing.
            Note: You may only use BYOL if you have existing licenses with rights to use these licenses in a third party cloud like AWS. For more information, see Prerequisites in the VM Import/Export User Guide.
            Valid values: AWS | BYOL
            

    :type Hypervisor: string
    :param Hypervisor: The target hypervisor platform.
            Valid values: xen
            

    :type Architecture: string
    :param Architecture: The architecture of the virtual machine.
            Valid values: i386 | x86_64
            

    :type Platform: string
    :param Platform: The operating system of the virtual machine.
            Valid values: Windows | Linux
            

    :type ClientData: dict
    :param ClientData: The client-specific data.
            UploadStart (datetime) --The time that the disk upload starts.
            UploadEnd (datetime) --The time that the disk upload ends.
            UploadSize (float) --The size of the uploaded disk image, in GiB.
            Comment (string) --A user-defined comment about the disk upload.
            

    :type ClientToken: string
    :param ClientToken: The token to enable idempotency for VM import requests.

    :type RoleName: string
    :param RoleName: The name of the role to use when not using the default role, 'vmimport'.

    :rtype: dict
    :return: {
        'ImportTaskId': 'string',
        'Architecture': 'string',
        'LicenseType': 'string',
        'Platform': 'string',
        'Hypervisor': 'string',
        'Description': 'string',
        'SnapshotDetails': [
            {
                'DiskImageSize': 123.0,
                'Description': 'string',
                'Format': 'string',
                'Url': 'string',
                'UserBucket': {
                    'S3Bucket': 'string',
                    'S3Key': 'string'
                },
                'DeviceName': 'string',
                'SnapshotId': 'string',
                'Progress': 'string',
                'StatusMessage': 'string',
                'Status': 'string'
            },
        ],
        'ImageId': 'string',
        'Progress': 'string',
        'StatusMessage': 'string',
        'Status': 'string'
    }
    
    
    """
    pass