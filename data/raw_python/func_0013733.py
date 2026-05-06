def update_function_configuration(FunctionName=None, Role=None, Handler=None, Description=None, Timeout=None, MemorySize=None, VpcConfig=None, Environment=None, Runtime=None, DeadLetterConfig=None, KMSKeyArn=None, TracingConfig=None):
    """
    Updates the configuration parameters for the specified Lambda function by using the values provided in the request. You provide only the parameters you want to change. This operation must only be used on an existing Lambda function and cannot be used to update the function's code.
    If you are using the versioning feature, note this API will always update the $LATEST version of your Lambda function. For information about the versioning feature, see AWS Lambda Function Versioning and Aliases .
    This operation requires permission for the lambda:UpdateFunctionConfiguration action.
    See also: AWS API Documentation
    
    Examples
    This operation updates a Lambda function's configuration
    Expected Output:
    
    :example: response = client.update_function_configuration(
        FunctionName='string',
        Role='string',
        Handler='string',
        Description='string',
        Timeout=123,
        MemorySize=123,
        VpcConfig={
            'SubnetIds': [
                'string',
            ],
            'SecurityGroupIds': [
                'string',
            ]
        },
        Environment={
            'Variables': {
                'string': 'string'
            }
        },
        Runtime='nodejs'|'nodejs4.3'|'nodejs6.10'|'java8'|'python2.7'|'python3.6'|'dotnetcore1.0'|'nodejs4.3-edge',
        DeadLetterConfig={
            'TargetArn': 'string'
        },
        KMSKeyArn='string',
        TracingConfig={
            'Mode': 'Active'|'PassThrough'
        }
    )
    
    
    :type FunctionName: string
    :param FunctionName: [REQUIRED]
            The name of the Lambda function.
            You can specify a function name (for example, Thumbnail ) or you can specify Amazon Resource Name (ARN) of the function (for example, arn:aws:lambda:us-west-2:account-id:function:ThumbNail ). AWS Lambda also allows you to specify a partial ARN (for example, account-id:Thumbnail ). Note that the length constraint applies only to the ARN. If you specify only the function name, it is limited to 64 character in length.
            

    :type Role: string
    :param Role: The Amazon Resource Name (ARN) of the IAM role that Lambda will assume when it executes your function.

    :type Handler: string
    :param Handler: The function that Lambda calls to begin executing your function. For Node.js, it is the module-name.export value in your function.

    :type Description: string
    :param Description: A short user-defined function description. AWS Lambda does not use this value. Assign a meaningful description as you see fit.

    :type Timeout: integer
    :param Timeout: The function execution time at which AWS Lambda should terminate the function. Because the execution time has cost implications, we recommend you set this value based on your expected execution time. The default is 3 seconds.

    :type MemorySize: integer
    :param MemorySize: The amount of memory, in MB, your Lambda function is given. AWS Lambda uses this memory size to infer the amount of CPU allocated to your function. Your function use-case determines your CPU and memory requirements. For example, a database operation might need less memory compared to an image processing function. The default value is 128 MB. The value must be a multiple of 64 MB.

    :type VpcConfig: dict
    :param VpcConfig: If your Lambda function accesses resources in a VPC, you provide this parameter identifying the list of security group IDs and subnet IDs. These must belong to the same VPC. You must provide at least one security group and one subnet ID.
            SubnetIds (list) --A list of one or more subnet IDs in your VPC.
            (string) --
            SecurityGroupIds (list) --A list of one or more security groups IDs in your VPC.
            (string) --
            

    :type Environment: dict
    :param Environment: The parent object that contains your environment's configuration settings.
            Variables (dict) --The key-value pairs that represent your environment's configuration settings.
            (string) --
            (string) --
            
            

    :type Runtime: string
    :param Runtime: The runtime environment for the Lambda function.
            To use the Python runtime v3.6, set the value to 'python3.6'. To use the Python runtime v2.7, set the value to 'python2.7'. To use the Node.js runtime v6.10, set the value to 'nodejs6.10'. To use the Node.js runtime v4.3, set the value to 'nodejs4.3'. To use the Python runtime v3.6, set the value to 'python3.6'. To use the Python runtime v2.7, set the value to 'python2.7'.
            Note
            You can no longer downgrade to the v0.10.42 runtime version. This version will no longer be supported as of early 2017.
            

    :type DeadLetterConfig: dict
    :param DeadLetterConfig: The parent object that contains the target ARN (Amazon Resource Name) of an Amazon SQS queue or Amazon SNS topic.
            TargetArn (string) --The Amazon Resource Name (ARN) of an Amazon SQS queue or Amazon SNS topic you specify as your Dead Letter Queue (DLQ).
            

    :type KMSKeyArn: string
    :param KMSKeyArn: The Amazon Resource Name (ARN) of the KMS key used to encrypt your function's environment variables. If you elect to use the AWS Lambda default service key, pass in an empty string ('') for this parameter.

    :type TracingConfig: dict
    :param TracingConfig: The parent object that contains your function's tracing settings.
            Mode (string) --Can be either PassThrough or Active. If PassThrough, Lambda will only trace the request from an upstream service if it contains a tracing header with 'sampled=1'. If Active, Lambda will respect any tracing header it receives from an upstream service. If no tracing header is received, Lambda will call X-Ray for a tracing decision.
            

    :rtype: dict
    :return: {
        'FunctionName': 'string',
        'FunctionArn': 'string',
        'Runtime': 'nodejs'|'nodejs4.3'|'nodejs6.10'|'java8'|'python2.7'|'python3.6'|'dotnetcore1.0'|'nodejs4.3-edge',
        'Role': 'string',
        'Handler': 'string',
        'CodeSize': 123,
        'Description': 'string',
        'Timeout': 123,
        'MemorySize': 123,
        'LastModified': 'string',
        'CodeSha256': 'string',
        'Version': 'string',
        'VpcConfig': {
            'SubnetIds': [
                'string',
            ],
            'SecurityGroupIds': [
                'string',
            ],
            'VpcId': 'string'
        },
        'DeadLetterConfig': {
            'TargetArn': 'string'
        },
        'Environment': {
            'Variables': {
                'string': 'string'
            },
            'Error': {
                'ErrorCode': 'string',
                'Message': 'string'
            }
        },
        'KMSKeyArn': 'string',
        'TracingConfig': {
            'Mode': 'Active'|'PassThrough'
        }
    }
    
    
    :returns: 
    (string) --
    
    """
    pass