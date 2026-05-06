def register_task_with_maintenance_window(WindowId=None, Targets=None, TaskArn=None, ServiceRoleArn=None, TaskType=None, TaskParameters=None, Priority=None, MaxConcurrency=None, MaxErrors=None, LoggingInfo=None, ClientToken=None):
    """
    Adds a new task to a Maintenance Window.
    See also: AWS API Documentation
    
    
    :example: response = client.register_task_with_maintenance_window(
        WindowId='string',
        Targets=[
            {
                'Key': 'string',
                'Values': [
                    'string',
                ]
            },
        ],
        TaskArn='string',
        ServiceRoleArn='string',
        TaskType='RUN_COMMAND',
        TaskParameters={
            'string': {
                'Values': [
                    'string',
                ]
            }
        },
        Priority=123,
        MaxConcurrency='string',
        MaxErrors='string',
        LoggingInfo={
            'S3BucketName': 'string',
            'S3KeyPrefix': 'string',
            'S3Region': 'string'
        },
        ClientToken='string'
    )
    
    
    :type WindowId: string
    :param WindowId: [REQUIRED]
            The id of the Maintenance Window the task should be added to.
            

    :type Targets: list
    :param Targets: [REQUIRED]
            The targets (either instances or tags). Instances are specified using Key=instanceids,Values=instanceid1,instanceid2. Tags are specified using Key=tag name,Values=tag value.
            (dict) --An array of search criteria that targets instances using a Key,Value combination that you specify. Targets is required if you don't provide one or more instance IDs in the call.
            Key (string) --User-defined criteria for sending commands that target instances that meet the criteria. Key can be tag:Amazon EC2 tagor InstanceIds. For more information about how to send commands that target instances using Key,Value parameters, see Executing a Command Using Systems Manager Run Command .
            Values (list) --User-defined criteria that maps to Key. For example, if you specified tag:ServerRole, you could specify value:WebServer to execute a command on instances that include Amazon EC2 tags of ServerRole,WebServer. For more information about how to send commands that target instances using Key,Value parameters, see Executing a Command Using Systems Manager Run Command .
            (string) --
            
            

    :type TaskArn: string
    :param TaskArn: [REQUIRED]
            The ARN of the task to execute
            

    :type ServiceRoleArn: string
    :param ServiceRoleArn: [REQUIRED]
            The role that should be assumed when executing the task.
            

    :type TaskType: string
    :param TaskType: [REQUIRED]
            The type of task being registered.
            

    :type TaskParameters: dict
    :param TaskParameters: The parameters that should be passed to the task when it is executed.
            (string) --
            (dict) --Defines the values for a task parameter.
            Values (list) --This field contains an array of 0 or more strings, each 1 to 255 characters in length.
            (string) --
            
            

    :type Priority: integer
    :param Priority: The priority of the task in the Maintenance Window, the lower the number the higher the priority. Tasks in a Maintenance Window are scheduled in priority order with tasks that have the same priority scheduled in parallel.

    :type MaxConcurrency: string
    :param MaxConcurrency: [REQUIRED]
            The maximum number of targets this task can be run for in parallel.
            

    :type MaxErrors: string
    :param MaxErrors: [REQUIRED]
            The maximum number of errors allowed before this task stops being scheduled.
            

    :type LoggingInfo: dict
    :param LoggingInfo: A structure containing information about an Amazon S3 bucket to write instance-level logs to.
            S3BucketName (string) -- [REQUIRED]The name of an Amazon S3 bucket where execution logs are stored .
            S3KeyPrefix (string) --(Optional) The Amazon S3 bucket subfolder.
            S3Region (string) -- [REQUIRED]The region where the Amazon S3 bucket is located.
            

    :type ClientToken: string
    :param ClientToken: User-provided idempotency token.
            This field is autopopulated if not provided.
            

    :rtype: dict
    :return: {
        'WindowTaskId': 'string'
    }
    
    
    """
    pass