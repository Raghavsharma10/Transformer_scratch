def head_object(Bucket=None, IfMatch=None, IfModifiedSince=None, IfNoneMatch=None, IfUnmodifiedSince=None, Key=None, Range=None, VersionId=None, SSECustomerAlgorithm=None, SSECustomerKey=None, SSECustomerKeyMD5=None, RequestPayer=None, PartNumber=None):
    """
    The HEAD operation retrieves metadata from an object without returning the object itself. This operation is useful if you're only interested in an object's metadata. To use HEAD, you must have READ access to the object.
    See also: AWS API Documentation
    
    
    :example: response = client.head_object(
        Bucket='string',
        IfMatch='string',
        IfModifiedSince=datetime(2015, 1, 1),
        IfNoneMatch='string',
        IfUnmodifiedSince=datetime(2015, 1, 1),
        Key='string',
        Range='string',
        VersionId='string',
        SSECustomerAlgorithm='string',
        SSECustomerKey='string',
        RequestPayer='requester',
        PartNumber=123
    )
    
    
    :type Bucket: string
    :param Bucket: [REQUIRED]

    :type IfMatch: string
    :param IfMatch: Return the object only if its entity tag (ETag) is the same as the one specified, otherwise return a 412 (precondition failed).

    :type IfModifiedSince: datetime
    :param IfModifiedSince: Return the object only if it has been modified since the specified time, otherwise return a 304 (not modified).

    :type IfNoneMatch: string
    :param IfNoneMatch: Return the object only if its entity tag (ETag) is different from the one specified, otherwise return a 304 (not modified).

    :type IfUnmodifiedSince: datetime
    :param IfUnmodifiedSince: Return the object only if it has not been modified since the specified time, otherwise return a 412 (precondition failed).

    :type Key: string
    :param Key: [REQUIRED]

    :type Range: string
    :param Range: Downloads the specified range bytes of an object. For more information about the HTTP Range header, go to http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.35.

    :type VersionId: string
    :param VersionId: VersionId used to reference a specific version of the object.

    :type SSECustomerAlgorithm: string
    :param SSECustomerAlgorithm: Specifies the algorithm to use to when encrypting the object (e.g., AES256).

    :type SSECustomerKey: string
    :param SSECustomerKey: Specifies the customer-provided encryption key for Amazon S3 to use in encrypting data. This value is used to store the object and then it is discarded; Amazon does not store the encryption key. The key must be appropriate for use with the algorithm specified in the x-amz-server-side -encryption -customer-algorithm header.

    :type SSECustomerKeyMD5: string
    :param SSECustomerKeyMD5: Specifies the 128-bit MD5 digest of the encryption key according to RFC 1321. Amazon S3 uses this header for a message integrity check to ensure the encryption key was transmitted without error.   Please note that this parameter is automatically populated if it is not provided. Including this parameter is not required

    :type RequestPayer: string
    :param RequestPayer: Confirms that the requester knows that she or he will be charged for the request. Bucket owners need not specify this parameter in their requests. Documentation on downloading objects from requester pays buckets can be found at http://docs.aws.amazon.com/AmazonS3/latest/dev/ObjectsinRequesterPaysBuckets.html

    :type PartNumber: integer
    :param PartNumber: Part number of the object being read. This is a positive integer between 1 and 10,000. Effectively performs a 'ranged' HEAD request for the part specified. Useful querying about the size of the part and the number of parts in this object.

    :rtype: dict
    :return: {
        'DeleteMarker': True|False,
        'AcceptRanges': 'string',
        'Expiration': 'string',
        'Restore': 'string',
        'LastModified': datetime(2015, 1, 1),
        'ContentLength': 123,
        'ETag': 'string',
        'MissingMeta': 123,
        'VersionId': 'string',
        'CacheControl': 'string',
        'ContentDisposition': 'string',
        'ContentEncoding': 'string',
        'ContentLanguage': 'string',
        'ContentType': 'string',
        'Expires': datetime(2015, 1, 1),
        'WebsiteRedirectLocation': 'string',
        'ServerSideEncryption': 'AES256'|'aws:kms',
        'Metadata': {
            'string': 'string'
        },
        'SSECustomerAlgorithm': 'string',
        'SSECustomerKeyMD5': 'string',
        'SSEKMSKeyId': 'string',
        'StorageClass': 'STANDARD'|'REDUCED_REDUNDANCY'|'STANDARD_IA',
        'RequestCharged': 'requester',
        'ReplicationStatus': 'COMPLETE'|'PENDING'|'FAILED'|'REPLICA',
        'PartsCount': 123
    }
    
    
    :returns: 
    (dict) --
    DeleteMarker (boolean) -- Specifies whether the object retrieved was (true) or was not (false) a Delete Marker. If false, this response header does not appear in the response.
    AcceptRanges (string) --
    Expiration (string) -- If the object expiration is configured (see PUT Bucket lifecycle), the response includes this header. It includes the expiry-date and rule-id key value pairs providing object expiration information. The value of the rule-id is URL encoded.
    Restore (string) -- Provides information about object restoration operation and expiration time of the restored object copy.
    LastModified (datetime) -- Last modified date of the object
    ContentLength (integer) -- Size of the body in bytes.
    ETag (string) -- An ETag is an opaque identifier assigned by a web server to a specific version of a resource found at a URL
    MissingMeta (integer) -- This is set to the number of metadata entries not returned in x-amz-meta headers. This can happen if you create metadata using an API like SOAP that supports more flexible metadata than the REST API. For example, using SOAP, you can create metadata whose values are not legal HTTP headers.
    VersionId (string) -- Version of the object.
    CacheControl (string) -- Specifies caching behavior along the request/reply chain.
    ContentDisposition (string) -- Specifies presentational information for the object.
    ContentEncoding (string) -- Specifies what content encodings have been applied to the object and thus what decoding mechanisms must be applied to obtain the media-type referenced by the Content-Type header field.
    ContentLanguage (string) -- The language the content is in.
    ContentType (string) -- A standard MIME type describing the format of the object data.
    Expires (datetime) -- The date and time at which the object is no longer cacheable.
    WebsiteRedirectLocation (string) -- If the bucket is configured as a website, redirects requests for this object to another object in the same bucket or to an external URL. Amazon S3 stores the value of this header in the object metadata.
    ServerSideEncryption (string) -- The Server-side encryption algorithm used when storing this object in S3 (e.g., AES256, aws:kms).
    Metadata (dict) -- A map of metadata to store with the object in S3.
    (string) --
    (string) --
    
    
    
    
    SSECustomerAlgorithm (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header confirming the encryption algorithm used.
    SSECustomerKeyMD5 (string) -- If server-side encryption with a customer-provided encryption key was requested, the response will include this header to provide round trip message integrity verification of the customer-provided encryption key.
    SSEKMSKeyId (string) -- If present, specifies the ID of the AWS Key Management Service (KMS) master encryption key that was used for the object.
    StorageClass (string) --
    RequestCharged (string) -- If present, indicates that the requester was successfully charged for the request.
    ReplicationStatus (string) --
    PartsCount (integer) -- The count of parts this object has.
    
    
    
    """
    pass