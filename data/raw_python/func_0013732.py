def create_function(FunctionName=None, Runtime=None, Role=None, Handler=None, Code=None, Description=None, Timeout=None, MemorySize=None, Publish=None, VpcConfig=None, DeadLetterConfig=None, Environment=None, KMSKeyArn=None, TracingConfig=None, Tags=None):
    """
    Creates a new Lambda function. The function metadata is created from the request parameters, and the code for the function is provided by a .zip file in the request body. If the function name already exists, the operation will fail. Note that the function name is case-sensitive.
    If you are using versioning, you can also publish a version of the Lambda function you are creating using the Publish parameter. For more information about versioning, see AWS Lambda Function Versioning and Aliases .
    This operation requires permission for the lambda:CreateFunction action.
    See also: AWS API Documentation
    
    Examples
    This example creates a Lambda function.
    Expected Output:
    
    :example: response = client.create_function(
        FunctionName='string',
        Runtime='nodejs'|'nodejs4.3'|'nodejs6.10'|'java8'|'python2.7'|'python3.6'|'dotnetcore1.0'|'nodejs4.3-edge',
        Role='string',
        Handler='string',
        Code={
            'ZipFile': b'bytes',
            'S3Bucket': 'string',
            'S3Key': 'string',
            'S3ObjectVersion': 'string'
        },
        Description='string',
        Timeout=123,
        MemorySize=123,
        Publish=True|False,
        VpcConfig={
            'SubnetIds': [
                'string',
            ],
            'SecurityGroupIds': [
                'string',
            ]
        },
        DeadLetterConfig={
            'TargetArn': 'string'
        },
        Environment={
            'Variables': {
                'string': 'string'
            }
        },
        KMSKeyArn='string',
        TracingConfig={
            'Mode': 'Active'|'PassThrough'
        },
        Tags={
            'string': 'string'
        }
    )
    
    
    :type FunctionName: string
    :param FunctionName: [REQUIRED]
            The name you want to assign to the function you are uploading. The function names appear in the console and are returned in the ListFunctions API. Function names are used to specify functions to other AWS Lambda API operations, such as Invoke . Note that the length constraint applies only to the ARN. If you specify only the function name, it is limited to 64 characters in length.
            

    :type Runtime: string
    :param Runtime: [REQUIRED]
            The runtime environment for the Lambda function you are uploading.
            To use the Python runtime v3.6, set the value to 'python3.6'. To use the Python runtime v2.7, set the value to 'python2.7'. To use the Node.js runtime v6.10, set the value to 'nodejs6.10'. To use the Node.js runtime v4.3, set the value to 'nodejs4.3'.
            Note
            You can no longer create functions using the v0.10.42 runtime version as of November, 2016. Existing functions will be supported until early 2017, but we recommend you migrate them to either nodejs6.10 or nodejs4.3 runtime version as soon as possible.
            

    :type Role: string
    :param Role: [REQUIRED]
            The Amazon Resource Name (ARN) of the IAM role that Lambda assumes when it executes your function to access any other Amazon Web Services (AWS) resources. For more information, see AWS Lambda: How it Works .
            

    :type Handler: string
    :param Handler: [REQUIRED]
            The function within your code that Lambda calls to begin execution. For Node.js, it is the module-name .*export* value in your function. For Java, it can be package.class-name::handler or package.class-name . For more information, see Lambda Function Handler (Java) .
            

    :type Code: dict
    :param Code: [REQUIRED]
            The code for the Lambda function.
            ZipFile (bytes) --The contents of your zip file containing your deployment package. If you are using the web API directly, the contents of the zip file must be base64-encoded. If you are using the AWS SDKs or the AWS CLI, the SDKs or CLI will do the encoding for you. For more information about creating a .zip file, see Execution Permissions in the AWS Lambda Developer Guide .
            S3Bucket (string) --Amazon S3 bucket name where the .zip file containing your deployment package is stored. This bucket must reside in the same AWS region where you are creating the Lambda function.
            S3Key (string) --The Amazon S3 object (the deployment package) key name you want to upload.
            S3ObjectVersion (string) --The Amazon S3 object (the deployment package) version you want to upload.
            

    :type Description: string
    :param Description: A short, user-defined function description. Lambda does not use this value. Assign a meaningful description as you see fit.

    :type Timeout: integer
    :param Timeout: The function execution time at which Lambda should terminate the function. Because the execution time has cost implications, we recommend you set this value based on your expected execution time. The default is 3 seconds.

    :type MemorySize: integer
    :param MemorySize: The amount of memory, in MB, your Lambda function is given. Lambda uses this memory size to infer the amount of CPU and memory allocated to your function. Your function use-case determines your CPU and memory requirements. For example, a database operation might need less memory compared to an image processing function. The default value is 128 MB. The value must be a multiple of 64 MB.

    :type Publish: boolean
    :param Publish: This boolean parameter can be used to request AWS Lambda to create the Lambda function and publish a version as an atomic operation.

    :type VpcConfig: dict
    :param VpcConfig: If your Lambda function accesses resources in a VPC, you provide this parameter identifying the list of security group IDs and subnet IDs. These must belong to the same VPC. You must provide at least one security group and one subnet ID.
            SubnetIds (list) --A list of one or more subnet IDs in your VPC.
            (string) --
            SecurityGroupIds (list) --A list of one or more security groups IDs in your VPC.
            (string) --
            

    :type DeadLetterConfig: dict
    :param DeadLetterConfig: The parent object that contains the target ARN (Amazon Resource Name) of an Amazon SQS queue or Amazon SNS topic.
            TargetArn (string) --The Amazon Resource Name (ARN) of an Amazon SQS queue or Amazon SNS topic you specify as your Dead Letter Queue (DLQ).
            

    :type Environment: dict
    :param Environment: The parent object that contains your environment's configuration settings.
            Variables (dict) --The key-value pairs that represent your environment's configuration settings.
            (string) --
            (string) --
            
            

    :type KMSKeyArn: string
    :param KMSKeyArn: The Amazon Resource Name (ARN) of the KMS key used to encrypt your function's environment variables. If not provided, AWS Lambda will use a default service key.

    :type TracingConfig: dict
    :param TracingConfig: The parent object that contains your function's tracing settings.
            Mode (string) --Can be either PassThrough or Active. If PassThrough, Lambda will only trace the request from an upstream service if it contains a tracing header with 'sampled=1'. If Active, Lambda will respect any tracing header it receives from an upstream service. If no tracing header is received, Lambda will call X-Ray for a tracing decision.
            

    :type Tags: dict
    :param Tags: The list of tags (key-value pairs) assigned to the new function.
            (string) --
            (string) --
            

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