def list_objects_v2(Bucket=None, Delimiter=None, EncodingType=None, MaxKeys=None, Prefix=None, ContinuationToken=None, FetchOwner=None, StartAfter=None, RequestPayer=None):
    """
    Returns some or all (up to 1000) of the objects in a bucket. You can use the request parameters as selection criteria to return a subset of the objects in a bucket. Note: ListObjectsV2 is the revised List Objects API and we recommend you use this revised API for new application development.
    See also: AWS API Documentation
    
    
    :example: response = client.list_objects_v2(
        Bucket='string',
        Delimiter='string',
        EncodingType='url',
        MaxKeys=123,
        Prefix='string',
        ContinuationToken='string',
        FetchOwner=True|False,
        StartAfter='string',
        RequestPayer='requester'
    )
    
    
    :type Bucket: string
    :param Bucket: [REQUIRED] Name of the bucket to list.

    :type Delimiter: string
    :param Delimiter: A delimiter is a character you use to group keys.

    :type EncodingType: string
    :param EncodingType: Encoding type used by Amazon S3 to encode object keys in the response.

    :type MaxKeys: integer
    :param MaxKeys: Sets the maximum number of keys returned in the response. The response might contain fewer keys but will never contain more.

    :type Prefix: string
    :param Prefix: Limits the response to keys that begin with the specified prefix.

    :type ContinuationToken: string
    :param ContinuationToken: ContinuationToken indicates Amazon S3 that the list is being continued on this bucket with a token. ContinuationToken is obfuscated and is not a real key

    :type FetchOwner: boolean
    :param FetchOwner: The owner field is not present in listV2 by default, if you want to return owner field with each key in the result then set the fetch owner field to true

    :type StartAfter: string
    :param StartAfter: StartAfter is where you want Amazon S3 to start listing from. Amazon S3 starts listing after this specified key. StartAfter can be any key in the bucket

    :type RequestPayer: string
    :param RequestPayer: Confirms that the requester knows that she or he will be charged for the list objects request in V2 style. Bucket owners need not specify this parameter in their requests.

    :rtype: dict
    :return: {
        'IsTruncated': True|False,
        'Contents': [
            {
                'Key': 'string',
                'LastModified': datetime(2015, 1, 1),
                'ETag': 'string',
                'Size': 123,
                'StorageClass': 'STANDARD'|'REDUCED_REDUNDANCY'|'GLACIER',
                'Owner': {
                    'DisplayName': 'string',
                    'ID': 'string'
                }
            },
        ],
        'Name': 'string',
        'Prefix': 'string',
        'Delimiter': 'string',
        'MaxKeys': 123,
        'CommonPrefixes': [
            {
                'Prefix': 'string'
            },
        ],
        'EncodingType': 'url',
        'KeyCount': 123,
        'ContinuationToken': 'string',
        'NextContinuationToken': 'string',
        'StartAfter': 'string'
    }
    
    
    :returns: 
    (dict) --
    IsTruncated (boolean) -- A flag that indicates whether or not Amazon S3 returned all of the results that satisfied the search criteria.
    Contents (list) -- Metadata about each object returned.
    (dict) --
    Key (string) --
    LastModified (datetime) --
    ETag (string) --
    Size (integer) --
    StorageClass (string) -- The class of storage used to store the object.
    Owner (dict) --
    DisplayName (string) --
    ID (string) --
    
    
    
    
    
    
    Name (string) -- Name of the bucket to list.
    Prefix (string) -- Limits the response to keys that begin with the specified prefix.
    Delimiter (string) -- A delimiter is a character you use to group keys.
    MaxKeys (integer) -- Sets the maximum number of keys returned in the response. The response might contain fewer keys but will never contain more.
    CommonPrefixes (list) -- CommonPrefixes contains all (if there are any) keys between Prefix and the next occurrence of the string specified by delimiter
    (dict) --
    Prefix (string) --
    
    
    
    
    EncodingType (string) -- Encoding type used by Amazon S3 to encode object keys in the response.
    KeyCount (integer) -- KeyCount is the number of keys returned with this request. KeyCount will always be less than equals to MaxKeys field. Say you ask for 50 keys, your result will include less than equals 50 keys
    ContinuationToken (string) -- ContinuationToken indicates Amazon S3 that the list is being continued on this bucket with a token. ContinuationToken is obfuscated and is not a real key
    NextContinuationToken (string) -- NextContinuationToken is sent when isTruncated is true which means there are more keys in the bucket that can be listed. The next list requests to Amazon S3 can be continued with this NextContinuationToken. NextContinuationToken is obfuscated and is not a real key
    StartAfter (string) -- StartAfter is where you want Amazon S3 to start listing from. Amazon S3 starts listing after this specified key. StartAfter can be any key in the bucket
    
    
    
    """
    pass