def put_object_acl(ACL=None, AccessControlPolicy=None, Bucket=None, GrantFullControl=None, GrantRead=None, GrantReadACP=None, GrantWrite=None, GrantWriteACP=None, Key=None, RequestPayer=None, VersionId=None):
    """
    uses the acl subresource to set the access control list (ACL) permissions for an object that already exists in a bucket
    See also: AWS API Documentation
    
    
    :example: response = client.put_object_acl(
        ACL='private'|'public-read'|'public-read-write'|'authenticated-read'|'aws-exec-read'|'bucket-owner-read'|'bucket-owner-full-control',
        AccessControlPolicy={
            'Grants': [
                {
                    'Grantee': {
                        'DisplayName': 'string',
                        'EmailAddress': 'string',
                        'ID': 'string',
                        'Type': 'CanonicalUser'|'AmazonCustomerByEmail'|'Group',
                        'URI': 'string'
                    },
                    'Permission': 'FULL_CONTROL'|'WRITE'|'WRITE_ACP'|'READ'|'READ_ACP'
                },
            ],
            'Owner': {
                'DisplayName': 'string',
                'ID': 'string'
            }
        },
        Bucket='string',
        GrantFullControl='string',
        GrantRead='string',
        GrantReadACP='string',
        GrantWrite='string',
        GrantWriteACP='string',
        Key='string',
        RequestPayer='requester',
        VersionId='string'
    )
    
    
    :type ACL: string
    :param ACL: The canned ACL to apply to the object.

    :type AccessControlPolicy: dict
    :param AccessControlPolicy: 
            Grants (list) -- A list of grants.
            (dict) --
            Grantee (dict) --
            DisplayName (string) -- Screen name of the grantee.
            EmailAddress (string) -- Email address of the grantee.
            ID (string) -- The canonical user ID of the grantee.
            Type (string) -- [REQUIRED] Type of grantee
            URI (string) -- URI of the grantee group.
            Permission (string) -- Specifies the permission given to the grantee.
            
            Owner (dict) --
            DisplayName (string) --
            ID (string) --
            

    :type Bucket: string
    :param Bucket: [REQUIRED]

    :type GrantFullControl: string
    :param GrantFullControl: Allows grantee the read, write, read ACP, and write ACP permissions on the bucket.

    :type GrantRead: string
    :param GrantRead: Allows grantee to list the objects in the bucket.

    :type GrantReadACP: string
    :param GrantReadACP: Allows grantee to read the bucket ACL.

    :type GrantWrite: string
    :param GrantWrite: Allows grantee to create, overwrite, and delete any object in the bucket.

    :type GrantWriteACP: string
    :param GrantWriteACP: Allows grantee to write the ACL for the applicable bucket.

    :type Key: string
    :param Key: [REQUIRED]

    :type RequestPayer: string
    :param RequestPayer: Confirms that the requester knows that she or he will be charged for the request. Bucket owners need not specify this parameter in their requests. Documentation on downloading objects from requester pays buckets can be found at http://docs.aws.amazon.com/AmazonS3/latest/dev/ObjectsinRequesterPaysBuckets.html

    :type VersionId: string
    :param VersionId: VersionId used to reference a specific version of the object.

    :rtype: dict
    :return: {
        'RequestCharged': 'requester'
    }
    
    
    :returns: 
    (dict) --
    RequestCharged (string) -- If present, indicates that the requester was successfully charged for the request.
    
    
    
    """
    pass