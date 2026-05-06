def create_nfs_file_share(ClientToken=None, NFSFileShareDefaults=None, GatewayARN=None, KMSEncrypted=None, KMSKey=None, Role=None, LocationARN=None, DefaultStorageClass=None, ClientList=None, Squash=None, ReadOnly=None):
    """
    Creates a file share on an existing file gateway. In Storage Gateway, a file share is a file system mount point backed by Amazon S3 cloud storage. Storage Gateway exposes file shares using a Network File System (NFS) interface. This operation is only supported in the file gateway architecture.
    See also: AWS API Documentation
    
    
    :example: response = client.create_nfs_file_share(
        ClientToken='string',
        NFSFileShareDefaults={
            'FileMode': 'string',
            'DirectoryMode': 'string',
            'GroupId': 123,
            'OwnerId': 123
        },
        GatewayARN='string',
        KMSEncrypted=True|False,
        KMSKey='string',
        Role='string',
        LocationARN='string',
        DefaultStorageClass='string',
        ClientList=[
            'string',
        ],
        Squash='string',
        ReadOnly=True|False
    )
    
    
    :type ClientToken: string
    :param ClientToken: [REQUIRED]
            A unique string value that you supply that is used by file gateway to ensure idempotent file share creation.
            

    :type NFSFileShareDefaults: dict
    :param NFSFileShareDefaults: File share default values. Optional.
            FileMode (string) --The Unix file mode in the form 'nnnn'. For example, '0666' represents the default file mode inside the file share. The default value is 0666.
            DirectoryMode (string) --The Unix directory mode in the form 'nnnn'. For example, '0666' represents the default access mode for all directories inside the file share. The default value is 0777.
            GroupId (integer) --The default group ID for the file share (unless the files have another group ID specified). The default value is nfsnobody.
            OwnerId (integer) --The default owner ID for files in the file share (unless the files have another owner ID specified). The default value is nfsnobody.
            

    :type GatewayARN: string
    :param GatewayARN: [REQUIRED]
            The Amazon Resource Name (ARN) of the file gateway on which you want to create a file share.
            

    :type KMSEncrypted: boolean
    :param KMSEncrypted: True to use Amazon S3 server side encryption with your own AWS KMS key, or false to use a key managed by Amazon S3. Optional.

    :type KMSKey: string
    :param KMSKey: The KMS key used for Amazon S3 server side encryption. This value can only be set when KmsEncrypted is true. Optional.

    :type Role: string
    :param Role: [REQUIRED]
            The ARN of the AWS Identity and Access Management (IAM) role that a file gateway assumes when it accesses the underlying storage.
            

    :type LocationARN: string
    :param LocationARN: [REQUIRED]
            The ARN of the backed storage used for storing file data.
            

    :type DefaultStorageClass: string
    :param DefaultStorageClass: The default storage class for objects put into an Amazon S3 bucket by file gateway. Possible values are S3_STANDARD or S3_STANDARD_IA. If this field is not populated, the default value S3_STANDARD is used. Optional.

    :type ClientList: list
    :param ClientList: The list of clients that are allowed to access the file gateway. The list must contain either valid IP addresses or valid CIDR blocks.
            (string) --
            

    :type Squash: string
    :param Squash: Maps a user to anonymous user. Valid options are the following:
            'RootSquash' - Only root is mapped to anonymous user.
            'NoSquash' - No one is mapped to anonymous user.
            'AllSquash' - Everyone is mapped to anonymous user.
            

    :type ReadOnly: boolean
    :param ReadOnly: Sets the write status of a file share: 'true' if the write status is read-only, and otherwise 'false'.

    :rtype: dict
    :return: {
        'FileShareARN': 'string'
    }
    
    
    """
    pass