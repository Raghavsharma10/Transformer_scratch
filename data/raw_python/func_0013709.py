def create_multipart_upload(ACL=None, Bucket=None, CacheControl=None, ContentDisposition=None, ContentEncoding=None, ContentLanguage=None, ContentType=None, Expires=None, GrantFullControl=None, GrantRead=None, GrantReadACP=None, GrantWriteACP=None, Key=None, Metadata=None, ServerSideEncryption=None, StorageClass=None, WebsiteRedirectLocation=None, SSECustomerAlgorithm=None, SSECustomerKey=None, SSECustomerKeyMD5=None, SSEKMSKeyId=None, RequestPayer=None):
    """
    Initiates a multipart upload and returns an upload ID.
    Note: After you initiate multipart upload and upload one or more parts, you must either complete or abort multipart upload in order to stop getting charged for storage of the uploaded parts. Only after you either complete or abort multipart upload, Amazon S3 frees up the parts storage and stops charging you for the parts storage.
    See also: AWS API Documentation
    
    
    :example: response = client.create_multipart_upload(
        ACL='private'|'public-read'|'public-read-write'|'authenticated-read'|'aws-exec-read'|'bucket-owner-read'|'bucket-owner-full-control',
        Bucket='string',
        CacheControl='string',
        ContentDisposition='string',
        ContentEncoding='string',
        ContentLanguage='string',
        ContentType='string',
        Expires=datetime(2015, 1, 1),
        GrantFullControl='string',
        GrantRead='string',
        GrantReadACP='string',
        GrantWriteACP='string',
        Key='string',
        Metadata={
            'string': 'string'
        },
        ServerSideEncryption='AES256'|'aws:kms',
        StorageClass='STANDARD'|'REDUCED_REDUNDANCY'|'STANDARD_IA',
        WebsiteRedirectLocation='string',
        SSECustomerAlgorithm='string',
        SSECustomerKey='string',
        SSEKMSKeyId='string',
        RequestPayer='requester'
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

    :type RequestPayer: string
    :param RequestPayer: Confirms that the requester knows that she or he will be charged for the request. Bucket owners need not specify this parameter in their requests. Documentation on downloading objects from requester pays buckets can be found at http://docs.aws.amazon.com/AmazonS3/latest/dev/ObjectsinRequesterPaysBuckets.html

    :rtype: dict
    :return: {
        'AbortDate': datetime(2015, 1, 1),
        'AbortRuleId': 'string',
        'Bucket': 'string',
        'Key': 'string',
        'UploadId': 'string',
        'ServerSideEncryption': 'AES256'|'aws:kms',
        'SSECustomerAlgorithm': 'string',
        'SSECustomerKeyMD5': 'string',
        'SSEKMSKeyId': 'string',
        'RequestCharged': 'requester'
    }
    
    
    :returns: 
    (dict) --
    AbortDate (datetime) -- Date when multipart upload will become eligible for abort operation by lifecycle.
    AbortRuleId (string) -- Id of the lifecycle rule that makes a multipart upload eligible for abort operation.
    Bucket (string) -- Name of the bucket to which the multipart upload was initiated.
    Key (string) -- Object key for which the multipart upload was initiated.
    UploadId (string) -- ID for the initiated multipart upload.
    ServerSideEncryption (string) -- The Server-side encryption algorithm used when storing this object in S3 (e.g., AES256, aws:kms).
    SSECustomerAlgorithm (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header confirming the encryption algorithm used.
    SSECustomerKeyMD5 (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header to provide round trip message integrity verification of the customer-provided encryption key.
    SSEKMSKeyId (string) -- If present, specifies the ID of the AWS Key Management Service (KMS) master encryption key that was used for the object.
    RequestCharged (string) -- If present, indicates that the requester was successfully charged for the request.
    
    
    
    """
    pass