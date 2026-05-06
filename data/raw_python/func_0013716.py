def upload_part_copy(Bucket=None, CopySource=None, CopySourceIfMatch=None, CopySourceIfModifiedSince=None, CopySourceIfNoneMatch=None, CopySourceIfUnmodifiedSince=None, CopySourceRange=None, Key=None, PartNumber=None, UploadId=None, SSECustomerAlgorithm=None, SSECustomerKey=None, SSECustomerKeyMD5=None, CopySourceSSECustomerAlgorithm=None, CopySourceSSECustomerKey=None, CopySourceSSECustomerKeyMD5=None, RequestPayer=None):
    """
    Uploads a part by copying data from an existing object as data source.
    See also: AWS API Documentation
    
    
    :example: response = client.upload_part_copy(
        Bucket='string',
        CopySource='string' or {'Bucket': 'string', 'Key': 'string', 'VersionId': 'string'},
        CopySourceIfMatch='string',
        CopySourceIfModifiedSince=datetime(2015, 1, 1),
        CopySourceIfNoneMatch='string',
        CopySourceIfUnmodifiedSince=datetime(2015, 1, 1),
        CopySourceRange='string',
        Key='string',
        PartNumber=123,
        UploadId='string',
        SSECustomerAlgorithm='string',
        SSECustomerKey='string',
        CopySourceSSECustomerAlgorithm='string',
        CopySourceSSECustomerKey='string',
        RequestPayer='requester'
    )
    
    
    :type Bucket: string
    :param Bucket: [REQUIRED]

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

    :type CopySourceRange: string
    :param CopySourceRange: The range of bytes to copy from the source object. The range value must use the form bytes=first-last, where the first and last are the zero-based byte offsets to copy. For example, bytes=0-9 indicates that you want to copy the first ten bytes of the source. You can copy a range only if the source object is greater than 5 GB.

    :type Key: string
    :param Key: [REQUIRED]

    :type PartNumber: integer
    :param PartNumber: [REQUIRED] Part number of part being copied. This is a positive integer between 1 and 10,000.

    :type UploadId: string
    :param UploadId: [REQUIRED] Upload ID identifying the multipart upload whose part is being copied.

    :type SSECustomerAlgorithm: string
    :param SSECustomerAlgorithm: Specifies the algorithm to use to when encrypting the object (e.g., AES256).

    :type SSECustomerKey: string
    :param SSECustomerKey: Specifies the customer-provided encryption key for Amazon S3 to use in encrypting data. This value is used to store the object and then it is discarded; Amazon does not store the encryption key. The key must be appropriate for use with the algorithm specified in the x-amz-server-side -encryption -customer-algorithm header. This must be the same encryption key specified in the initiate multipart upload request.

    :type SSECustomerKeyMD5: string
    :param SSECustomerKeyMD5: Specifies the 128-bit MD5 digest of the encryption key according to RFC 1321. Amazon S3 uses this header for a message integrity check to ensure the encryption key was transmitted without error.   Please note that this parameter is automatically populated if it is not provided. Including this parameter is not required

    :type CopySourceSSECustomerAlgorithm: string
    :param CopySourceSSECustomerAlgorithm: Specifies the algorithm to use when decrypting the source object (e.g., AES256).

    :type CopySourceSSECustomerKey: string
    :param CopySourceSSECustomerKey: Specifies the customer-provided encryption key for Amazon S3 to use to decrypt the source object. The encryption key provided in this header must be one that was used when the source object was created.

    :type CopySourceSSECustomerKeyMD5: string
    :param CopySourceSSECustomerKeyMD5: Specifies the 128-bit MD5 digest of the encryption key according to RFC 1321. Amazon S3 uses this header for a message integrity check to ensure the encryption key was transmitted without error.   Please note that this parameter is automatically populated if it is not provided. Including this parameter is not required

    :type RequestPayer: string
    :param RequestPayer: Confirms that the requester knows that she or he will be charged for the request. Bucket owners need not specify this parameter in their requests. Documentation on downloading objects from requester pays buckets can be found at http://docs.aws.amazon.com/AmazonS3/latest/dev/ObjectsinRequesterPaysBuckets.html

    :rtype: dict
    :return: {
        'CopySourceVersionId': 'string',
        'CopyPartResult': {
            'ETag': 'string',
            'LastModified': datetime(2015, 1, 1)
        },
        'ServerSideEncryption': 'AES256'|'aws:kms',
        'SSECustomerAlgorithm': 'string',
        'SSECustomerKeyMD5': 'string',
        'SSEKMSKeyId': 'string',
        'RequestCharged': 'requester'
    }
    
    
    :returns: 
    (dict) --
    CopySourceVersionId (string) -- The version of the source object that was copied, if you have enabled versioning on the source bucket.
    CopyPartResult (dict) --
    ETag (string) -- Entity tag of the object.
    LastModified (datetime) -- Date and time at which the object was uploaded.
    
    
    ServerSideEncryption (string) -- The Server-side encryption algorithm used when storing this object in S3 (e.g., AES256, aws:kms).
    SSECustomerAlgorithm (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header confirming the encryption algorithm used.
    SSECustomerKeyMD5 (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header to provide round trip message integrity verification of the customer-provided encryption key.
    SSEKMSKeyId (string) -- If present, specifies the ID of the AWS Key Management Service (KMS) master encryption key that was used for the object.
    RequestCharged (string) -- If present, indicates that the requester was successfully charged for the request.
    
    
    
    """
    pass