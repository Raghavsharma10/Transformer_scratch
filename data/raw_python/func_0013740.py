def send_command(InstanceIds=None, Targets=None, DocumentName=None, DocumentHash=None, DocumentHashType=None, TimeoutSeconds=None, Comment=None, Parameters=None, OutputS3Region=None, OutputS3BucketName=None, OutputS3KeyPrefix=None, MaxConcurrency=None, MaxErrors=None, ServiceRoleArn=None, NotificationConfig=None):
    """
    Executes commands on one or more remote instances.
    See also: AWS API Documentation
    
    
    :example: response = client.send_command(
        InstanceIds=[
            'string',
        ],
        Targets=[
            {
                'Key': 'string',
                'Values': [
                    'string',
                ]
            },
        ],
        DocumentName='string',
        DocumentHash='string',
        DocumentHashType='Sha256'|'Sha1',
        TimeoutSeconds=123,
        Comment='string',
        Parameters={
            'string': [
                'string',
            ]
        },
        OutputS3Region='string',
        OutputS3BucketName='string',
        OutputS3KeyPrefix='string',
        MaxConcurrency='string',
        MaxErrors='string',
        ServiceRoleArn='string',
        NotificationConfig={
            'NotificationArn': 'string',
            'NotificationEvents': [
                'All'|'InProgress'|'Success'|'TimedOut'|'Cancelled'|'Failed',
            ],
            'NotificationType': 'Command'|'Invocation'
        }
    )
    
    
    :type InstanceIds: list
    :param InstanceIds: The instance IDs where the command should execute. You can specify a maximum of 50 IDs. If you prefer not to list individual instance IDs, you can instead send commands to a fleet of instances using the Targets parameter, which accepts EC2 tags.
            (string) --
            

    :type Targets: list
    :param Targets: (Optional) An array of search criteria that targets instances using a Key,Value combination that you specify. Targets is required if you don't provide one or more instance IDs in the call. For more information about how to use Targets, see Executing a Command Using Systems Manager Run Command .
            (dict) --An array of search criteria that targets instances using a Key,Value combination that you specify. Targets is required if you don't provide one or more instance IDs in the call.
            Key (string) --User-defined criteria for sending commands that target instances that meet the criteria. Key can be tag:Amazon EC2 tagor InstanceIds. For more information about how to send commands that target instances using Key,Value parameters, see Executing a Command Using Systems Manager Run Command .
            Values (list) --User-defined criteria that maps to Key. For example, if you specified tag:ServerRole, you could specify value:WebServer to execute a command on instances that include Amazon EC2 tags of ServerRole,WebServer. For more information about how to send commands that target instances using Key,Value parameters, see Executing a Command Using Systems Manager Run Command .
            (string) --
            
            

    :type DocumentName: string
    :param DocumentName: [REQUIRED]
            Required. The name of the Systems Manager document to execute. This can be a public document or a custom document.
            

    :type DocumentHash: string
    :param DocumentHash: The Sha256 or Sha1 hash created by the system when the document was created.
            Note
            Sha1 hashes have been deprecated.
            

    :type DocumentHashType: string
    :param DocumentHashType: Sha256 or Sha1.
            Note
            Sha1 hashes have been deprecated.
            

    :type TimeoutSeconds: integer
    :param TimeoutSeconds: If this time is reached and the command has not already started executing, it will not execute.

    :type Comment: string
    :param Comment: User-specified information about the command, such as a brief description of what the command should do.

    :type Parameters: dict
    :param Parameters: The required and optional parameters specified in the document being executed.
            (string) --
            (list) --
            (string) --
            
            

    :type OutputS3Region: string
    :param OutputS3Region: (Optional) The region where the Amazon Simple Storage Service (Amazon S3) output bucket is located. The default value is the region where Run Command is being called.

    :type OutputS3BucketName: string
    :param OutputS3BucketName: The name of the S3 bucket where command execution responses should be stored.

    :type OutputS3KeyPrefix: string
    :param OutputS3KeyPrefix: The directory structure within the S3 bucket where the responses should be stored.

    :type MaxConcurrency: string
    :param MaxConcurrency: (Optional) The maximum number of instances that are allowed to execute the command at the same time. You can specify a number such as 10 or a percentage such as 10%. The default value is 50. For more information about how to use MaxConcurrency, see Executing a Command Using Systems Manager Run Command .

    :type MaxErrors: string
    :param MaxErrors: The maximum number of errors allowed without the command failing. When the command fails one more time beyond the value of MaxErrors, the systems stops sending the command to additional targets. You can specify a number like 10 or a percentage like 10%. The default value is 50. For more information about how to use MaxErrors, see Executing a Command Using Systems Manager Run Command .

    :type ServiceRoleArn: string
    :param ServiceRoleArn: The IAM role that Systems Manager uses to send notifications.

    :type NotificationConfig: dict
    :param NotificationConfig: Configurations for sending notifications.
            NotificationArn (string) --An Amazon Resource Name (ARN) for a Simple Notification Service (SNS) topic. Run Command pushes notifications about command status changes to this topic.
            NotificationEvents (list) --The different events for which you can receive notifications. These events include the following: All (events), InProgress, Success, TimedOut, Cancelled, Failed. To learn more about these events, see Setting Up Events and Notifications in the Amazon EC2 Systems Manager User Guide .
            (string) --
            NotificationType (string) --Command: Receive notification when the status of a command changes. Invocation: For commands sent to multiple instances, receive notification on a per-instance basis when the status of a command changes.
            

    :rtype: dict
    :return: {
        'Command': {
            'CommandId': 'string',
            'DocumentName': 'string',
            'Comment': 'string',
            'ExpiresAfter': datetime(2015, 1, 1),
            'Parameters': {
                'string': [
                    'string',
                ]
            },
            'InstanceIds': [
                'string',
            ],
            'Targets': [
                {
                    'Key': 'string',
                    'Values': [
                        'string',
                    ]
                },
            ],
            'RequestedDateTime': datetime(2015, 1, 1),
            'Status': 'Pending'|'InProgress'|'Success'|'Cancelled'|'Failed'|'TimedOut'|'Cancelling',
            'StatusDetails': 'string',
            'OutputS3Region': 'string',
            'OutputS3BucketName': 'string',
            'OutputS3KeyPrefix': 'string',
            'MaxConcurrency': 'string',
            'MaxErrors': 'string',
            'TargetCount': 123,
            'CompletedCount': 123,
            'ErrorCount': 123,
            'ServiceRole': 'string',
            'NotificationConfig': {
                'NotificationArn': 'string',
                'NotificationEvents': [
                    'All'|'InProgress'|'Success'|'TimedOut'|'Cancelled'|'Failed',
                ],
                'NotificationType': 'Command'|'Invocation'
            }
        }
    }
    
    
    :returns: 
    (string) --
    (list) --
    (string) --
    
    
    
    
    
    """
    pass