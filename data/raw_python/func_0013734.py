def create_trail(Name=None, S3BucketName=None, S3KeyPrefix=None, SnsTopicName=None, IncludeGlobalServiceEvents=None, IsMultiRegionTrail=None, EnableLogFileValidation=None, CloudWatchLogsLogGroupArn=None, CloudWatchLogsRoleArn=None, KmsKeyId=None):
    """
    Creates a trail that specifies the settings for delivery of log data to an Amazon S3 bucket. A maximum of five trails can exist in a region, irrespective of the region in which they were created.
    See also: AWS API Documentation
    
    
    :example: response = client.create_trail(
        Name='string',
        S3BucketName='string',
        S3KeyPrefix='string',
        SnsTopicName='string',
        IncludeGlobalServiceEvents=True|False,
        IsMultiRegionTrail=True|False,
        EnableLogFileValidation=True|False,
        CloudWatchLogsLogGroupArn='string',
        CloudWatchLogsRoleArn='string',
        KmsKeyId='string'
    )
    
    
    :type Name: string
    :param Name: [REQUIRED]
            Specifies the name of the trail. The name must meet the following requirements:
            Contain only ASCII letters (a-z, A-Z), numbers (0-9), periods (.), underscores (_), or dashes (-)
            Start with a letter or number, and end with a letter or number
            Be between 3 and 128 characters
            Have no adjacent periods, underscores or dashes. Names like my-_namespace and my--namespace are invalid.
            Not be in IP address format (for example, 192.168.5.4)
            

    :type S3BucketName: string
    :param S3BucketName: [REQUIRED]
            Specifies the name of the Amazon S3 bucket designated for publishing log files. See Amazon S3 Bucket Naming Requirements .
            

    :type S3KeyPrefix: string
    :param S3KeyPrefix: Specifies the Amazon S3 key prefix that comes after the name of the bucket you have designated for log file delivery. For more information, see Finding Your CloudTrail Log Files . The maximum length is 200 characters.

    :type SnsTopicName: string
    :param SnsTopicName: Specifies the name of the Amazon SNS topic defined for notification of log file delivery. The maximum length is 256 characters.

    :type IncludeGlobalServiceEvents: boolean
    :param IncludeGlobalServiceEvents: Specifies whether the trail is publishing events from global services such as IAM to the log files.

    :type IsMultiRegionTrail: boolean
    :param IsMultiRegionTrail: Specifies whether the trail is created in the current region or in all regions. The default is false.

    :type EnableLogFileValidation: boolean
    :param EnableLogFileValidation: Specifies whether log file integrity validation is enabled. The default is false.
            Note
            When you disable log file integrity validation, the chain of digest files is broken after one hour. CloudTrail will not create digest files for log files that were delivered during a period in which log file integrity validation was disabled. For example, if you enable log file integrity validation at noon on January 1, disable it at noon on January 2, and re-enable it at noon on January 10, digest files will not be created for the log files delivered from noon on January 2 to noon on January 10. The same applies whenever you stop CloudTrail logging or delete a trail.
            

    :type CloudWatchLogsLogGroupArn: string
    :param CloudWatchLogsLogGroupArn: Specifies a log group name using an Amazon Resource Name (ARN), a unique identifier that represents the log group to which CloudTrail logs will be delivered. Not required unless you specify CloudWatchLogsRoleArn.

    :type CloudWatchLogsRoleArn: string
    :param CloudWatchLogsRoleArn: Specifies the role for the CloudWatch Logs endpoint to assume to write to a user's log group.

    :type KmsKeyId: string
    :param KmsKeyId: Specifies the KMS key ID to use to encrypt the logs delivered by CloudTrail. The value can be an alias name prefixed by 'alias/', a fully specified ARN to an alias, a fully specified ARN to a key, or a globally unique identifier.
            Examples:
            alias/MyAliasName
            arn:aws:kms:us-east-1:123456789012:alias/MyAliasName
            arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012
            12345678-1234-1234-1234-123456789012
            

    :rtype: dict
    :return: {
        'Name': 'string',
        'S3BucketName': 'string',
        'S3KeyPrefix': 'string',
        'SnsTopicName': 'string',
        'SnsTopicARN': 'string',
        'IncludeGlobalServiceEvents': True|False,
        'IsMultiRegionTrail': True|False,
        'TrailARN': 'string',
        'LogFileValidationEnabled': True|False,
        'CloudWatchLogsLogGroupArn': 'string',
        'CloudWatchLogsRoleArn': 'string',
        'KmsKeyId': 'string'
    }
    
    
    """
    pass