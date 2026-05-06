def copy_object(ACL=None, Bucket=None, CacheControl=None, ContentDisposition=None, ContentEncoding=None, ContentLanguage=None, ContentType=None, CopySource=None, CopySourceIfMatch=None, CopySourceIfModifiedSince=None, CopySourceIfNoneMatch=None, CopySourceIfUnmodifiedSince=None, Expires=None, GrantFullControl=None, GrantRead=None, GrantReadACP=None, GrantWriteACP=None, Key=None, Metadata=None, MetadataDirective=None, TaggingDirective=None, ServerSideEncryption=None, StorageClass=None, WebsiteRedirectLocation=None, SSECustomerAlgorithm=None, SSECustomerKey=None, SSECustomerKeyMD5=None, SSEKMSKeyId=None, CopySourceSSECustomerAlgorithm=None, CopySourceSSECustomerKey=None, CopySourceSSECustomerKeyMD5=None, RequestPayer=None, Tagging=None):
    """
    Creates a copy of an object that is already stored in Amazon S3.
    See also: AWS API Documentation
    
    
    :example: response = client.copy_object(
        ACL='private'|'public-read'|'public-read-write'|'authenticated-read'|'aws-exec-read'|'bucket-owner-read'|'bucket-owner-full-control',
        Bucket='string',
        CacheControl='string',
        ContentDisposition='string',
        ContentEncoding='string',
        ContentLanguage='string',
        ContentType='string',
        CopySource='string' or {'Bucket': 'string', 'Key': 'string', 'VersionId': 'string'},
        CopySourceIfMatch='string',
        CopySourceIfModifiedSince=datetime(2015, 1, 1),
        CopySourceIfNoneMatch='string',
        CopySourceIfUnmodifiedSince=datetime(2015, 1, 1),
        Expires=datetime(2015, 1, 1),
        GrantFullControl='string',
        GrantRead='string',
        GrantReadACP='string',
        GrantWriteACP='string',
        Key='string',
        Metadata={
            'string': 'string'
        },
        MetadataDirective='COPY'|'REPLACE',
        TaggingDirective='COPY'|'REPLACE',
        ServerSideEncryption='AES256'|'aws:kms',
        StorageClass='STANDARD'|'REDUCED_REDUNDANCY'|'STANDARD_IA',
        WebsiteRedirectLocation='string',
        SSECustomerAlgorithm='string',
        SSECustomerKey='string',
        SSEKMSKeyId='string',
        CopySourceSSECustomerAlgorithm='string',
        CopySourceSSECustomerKey='string',
        RequestPayer='requester',
        Tagging='string'
    )
    
    
    :type ACL: string
    :param ACL: The canned ACL to apply to the object.

    :type Bucket: string
    :param Bucket: [REQUIRED]

    :type CacheControl: string
    :param CacheControl: Specifies caching behavior along the request/reply chain.

    :type ContentDisposition: string
    :param ContentDisposition: Specifies presentational information for the object.

    :type ContentEncoding: string
    :param ContentEncoding: Specifies what content encodings have been applied to the object and thus what decoding mechanisms must be applied to obtain the media-type referenced by the Content-Type header field.

    :type ContentLanguage: string
    :param ContentLanguage: The language the content is in.

    :type ContentType: string
    :param ContentType: A standard MIME type describing the format of the object data.

    :type CopySource: str or dict
    :param CopySource: [REQUIRED] The name of the source bucket, key name of the source object, and optional version ID of the source object. You can either provide this value as a string or a dictionary. The string form is {bucket}/{key} or {bucket}/{key}?versionId={versionId} if you want to copy a specific version. You can also provide this value as a dictionary. The dictionary format is recommended over the string format because it is more explicit. The dictionary format is: {'Bucket': 'bucket', 'Key': 'key', 'VersionId': 'id'}. Note that the VersionId key is optional and may be omitted.

    :type CopySourceIfMatch: string
    :param CopySourceIfMatch: Copies the object if its entity tag (ETag) matches the specified tag.

    :type CopySourceIfModifiedSince: datetime
    :param CopySourceIfModifiedSince: Copies the object if it has been modified since the specified time.

    :type CopySourceIfNoneMatch: string
    :param CopySourceIfNoneMatch: Copies the object if its entity tag (ETag) is different than the specified ETag.

    :type CopySourceIfUnmodifiedSince: datetime
    :param CopySourceIfUnmodifiedSince: Copies the object if it hasn't been modified since the specified time.

    :type Expires: datetime
    :param Expires: The date and time at which the object is no longer cacheable.

    :type GrantFullControl: string
    :param GrantFullControl: Gives the grantee READ, READ_ACP, and WRITE_ACP permissions on the object.

    :type GrantRead: string
    :param GrantRead: Allows grantee to read the object data and its metadata.

    :type GrantReadACP: string
    :param GrantReadACP: Allows grantee to read the object ACL.

    :type GrantWriteACP: string
    :param GrantWriteACP: Allows grantee to write the ACL for the applicable object.

    :type Key: string
    :param Key: [REQUIRED]

    :type Metadata: dict
    :param Metadata: A map of metadata to store with the object in S3.
            (string) --
            (string) --
            

    :type MetadataDirective: string
    :param MetadataDirective: Specifies whether the metadata is copied from the source object or replaced with metadata provided in the request.

    :type TaggingDirective: string
    :param TaggingDirective: Specifies whether the object tag-set are copied from the source object or replaced with tag-set provided in the request.

    :type ServerSideEncryption: string
    :param ServerSideEncryption: The Server-side encryption algorithm used when storing this object in S3 (e.g., AES256, aws:kms).

    :type StorageClass: string
    :param StorageClass: The type of storage to use for the object. Defaults to 'STANDARD'.

    :type WebsiteRedirectLocation: string
    :param WebsiteRedirectLocation: If the bucket is configured as a website, redirects requests for this object to another object in the same bucket or to an external URL. Amazon S3 stores the value of this header in the object metadata.

    :type SSECustomerAlgorithm: string
    :param SSECustomerAlgorithm: Specifies the algorithm to use to when encrypting the object (e.g., AES256).

    :type SSECustomerKey: string
    :param SSECustomerKey: Specifies the customer-provided encryption key for Amazon S3 to use in encrypting data. This value is used to store the object and then it is discarded; Amazon does not store the encryption key. The key must be appropriate for use with the algorithm specified in the x-amz-server-side -encryption -customer-algorithm header.

    :type SSECustomerKeyMD5: string
    :param SSECustomerKeyMD5: Specifies the 128-bit MD5 digest of the encryption key according to RFC 1321. Amazon S3 uses this header for a message integrity check to ensure the encryption key was transmitted without error.   Please note that this parameter is automatically populated if it is not provided. Including this parameter is not required

    :type SSEKMSKeyId: string
    :param SSEKMSKeyId: Specifies the AWS KMS key ID to use for object encryption. All GET and PUT requests for an object protected by AWS KMS will fail if not made via SSL or using SigV4. Documentation on configuring any of the officially supported AWS SDKs and CLI can be found at http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingAWSSDK.html#specify-signature-version

    :type CopySourceSSECustomerAlgorithm: string
    :param CopySourceSSECustomerAlgorithm: Specifies the algorithm to use when decrypting the source object (e.g., AES256).

    :type CopySourceSSECustomerKey: string
    :param CopySourceSSECustomerKey: Specifies the customer-provided encryption key for Amazon S3 to use to decrypt the source object. The encryption key provided in this header must be one that was used when the source object was created.

    :type CopySourceSSECustomerKeyMD5: string
    :param CopySourceSSECustomerKeyMD5: Specifies the 128-bit MD5 digest of the encryption key according to RFC 1321. Amazon S3 uses this header for a message integrity check to ensure the encryption key was transmitted without error.   Please note that this parameter is automatically populated if it is not provided. Including this parameter is not required

    :type RequestPayer: string
    :param RequestPayer: Confirms that the requester knows that she or he will be charged for the request. Bucket owners need not specify this parameter in their requests. Documentation on downloading objects from requester pays buckets can be found at http://docs.aws.amazon.com/AmazonS3/latest/dev/ObjectsinRequesterPaysBuckets.html

    :type Tagging: string
    :param Tagging: The tag-set for the object destination object this value must be used in conjunction with the TaggingDirective. The tag-set must be encoded as URL Query parameters

    :rtype: dict
    :return: {
        'CopyObjectResult': {
            'ETag': 'string',
            'LastModified': datetime(2015, 1, 1)
        },
        'Expiration': 'string',
        'CopySourceVersionId': 'string',
        'VersionId': 'string',
        'ServerSideEncryption': 'AES256'|'aws:kms',
        'SSECustomerAlgorithm': 'string',
        'SSECustomerKeyMD5': 'string',
        'SSEKMSKeyId': 'string',
        'RequestCharged': 'requester'
    }
    
    
    :returns: 
    (dict) --
    CopyObjectResult (dict) --
    ETag (string) --
    LastModified (datetime) --
    
    
    Expiration (string) -- If the object expiration is configured, the response includes this header.
    CopySourceVersionId (string) --
    VersionId (string) -- Version ID of the newly created copy.
    ServerSideEncryption (string) -- The Server-side encryption algorithm used when storing this object in S3 (e.g., AES256, aws:kms).
    SSECustomerAlgorithm (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header confirming the encryption algorithm used.
    SSECustomerKeyMD5 (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header to provide round trip message integrity verification of the customer-provided encryption key.
    SSEKMSKeyId (string) -- If present, specifies the ID of the AWS Key Management Service (KMS) master encryption key that was used for the object.
    RequestCharged (string) -- If present, indicates that the requester was successfully charged for the request.
    
    
    
    """
    pass