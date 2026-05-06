def upload_part(Body=None, Bucket=None, ContentLength=None, ContentMD5=None, Key=None, PartNumber=None, UploadId=None, SSECustomerAlgorithm=None, SSECustomerKey=None, SSECustomerKeyMD5=None, RequestPayer=None):
    """
    Uploads a part in a multipart upload.
    Note: After you initiate multipart upload and upload one or more parts, you must either complete or abort multipart upload in order to stop getting charged for storage of the uploaded parts. Only after you either complete or abort multipart upload, Amazon S3 frees up the parts storage and stops charging you for the parts storage.
    See also: AWS API Documentation
    
    
    :example: response = client.upload_part(
        Body=b'bytes'|file,
        Bucket='string',
        ContentLength=123,
        ContentMD5='string',
        Key='string',
        PartNumber=123,
        UploadId='string',
        SSECustomerAlgorithm='string',
        SSECustomerKey='string',
        RequestPayer='requester'
    )
    
    
    :type Body: bytes or seekable file-like object
    :param Body: Object data.

    :type Bucket: string
    :param Bucket: [REQUIRED] Name of the bucket to which the multipart upload was initiated.

    :type ContentLength: integer
    :param ContentLength: Size of the body in bytes. This parameter is useful when the size of the body cannot be determined automatically.

    :type ContentMD5: string
    :param ContentMD5: The base64-encoded 128-bit MD5 digest of the part data.

    :type Key: string
    :param Key: [REQUIRED] Object key for which the multipart upload was initiated.

    :type PartNumber: integer
    :param PartNumber: [REQUIRED] Part number of part being uploaded. This is a positive integer between 1 and 10,000.

    :type UploadId: string
    :param UploadId: [REQUIRED] Upload ID identifying the multipart upload whose part is being uploaded.

    :type SSECustomerAlgorithm: string
    :param SSECustomerAlgorithm: Specifies the algorithm to use to when encrypting the object (e.g., AES256).

    :type SSECustomerKey: string
    :param SSECustomerKey: Specifies the customer-provided encryption key for Amazon S3 to use in encrypting data. This value is used to store the object and then it is discarded; Amazon does not store the encryption key. The key must be appropriate for use with the algorithm specified in the x-amz-server-side -encryption -customer-algorithm header. This must be the same encryption key specified in the initiate multipart upload request.

    :type SSECustomerKeyMD5: string
    :param SSECustomerKeyMD5: Specifies the 128-bit MD5 digest of the encryption key according to RFC 1321. Amazon S3 uses this header for a message integrity check to ensure the encryption key was transmitted without error.   Please note that this parameter is automatically populated if it is not provided. Including this parameter is not required

    :type RequestPayer: string
    :param RequestPayer: Confirms that the requester knows that she or he will be charged for the request. Bucket owners need not specify this parameter in their requests. Documentation on downloading objects from requester pays buckets can be found at http://docs.aws.amazon.com/AmazonS3/latest/dev/ObjectsinRequesterPaysBuckets.html

    :rtype: dict
    :return: {
        'ServerSideEncryption': 'AES256'|'aws:kms',
        'ETag': 'string',
        'SSECustomerAlgorithm': 'string',
        'SSECustomerKeyMD5': 'string',
        'SSEKMSKeyId': 'string',
        'RequestCharged': 'requester'
    }
    
    
    :returns: 
    (dict) --
    ServerSideEncryption (string) -- The Server-side encryption algorithm used when storing this object in S3 (e.g., AES256, aws:kms).
    ETag (string) -- Entity tag for the uploaded object.
    SSECustomerAlgorithm (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header confirming the encryption algorithm used.
    SSECustomerKeyMD5 (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header to provide round trip message integrity verification of the customer-provided encryption key.
    SSEKMSKeyId (string) -- If present, specifies the ID of the AWS Key Management Service (KMS) master encryption key that was used for the object.
    RequestCharged (string) -- If present, indicates that the requester was successfully charged for the request.
    
    
    
    """
    pass