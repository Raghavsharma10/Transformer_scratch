def start_workflow_execution(domain=None, workflowId=None, workflowType=None, taskList=None, taskPriority=None, input=None, executionStartToCloseTimeout=None, tagList=None, taskStartToCloseTimeout=None, childPolicy=None, lambdaRole=None):
    """
    Starts an execution of the workflow type in the specified domain using the provided workflowId and input data.
    This action returns the newly started workflow execution.
    Access Control
    You can use IAM policies to control this action's access to Amazon SWF resources as follows:
    If the caller does not have sufficient permissions to invoke the action, or the parameter values fall outside the specified constraints, the action fails. The associated event attribute's cause parameter will be set to OPERATION_NOT_PERMITTED. For details and example IAM policies, see Using IAM to Manage Access to Amazon SWF Workflows .
    See also: AWS API Documentation
    
    
    :example: response = client.start_workflow_execution(
        domain='string',
        workflowId='string',
        workflowType={
            'name': 'string',
            'version': 'string'
        },
        taskList={
            'name': 'string'
        },
        taskPriority='string',
        input='string',
        executionStartToCloseTimeout='string',
        tagList=[
            'string',
        ],
        taskStartToCloseTimeout='string',
        childPolicy='TERMINATE'|'REQUEST_CANCEL'|'ABANDON',
        lambdaRole='string'
    )
    
    
    :type domain: string
    :param domain: [REQUIRED]
            The name of the domain in which the workflow execution is created.
            

    :type workflowId: string
    :param workflowId: [REQUIRED]
            The user defined identifier associated with the workflow execution. You can use this to associate a custom identifier with the workflow execution. You may specify the same identifier if a workflow execution is logically a restart of a previous execution. You cannot have two open workflow executions with the same workflowId at the same time.
            The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
            

    :type workflowType: dict
    :param workflowType: [REQUIRED]
            The type of the workflow to start.
            name (string) -- [REQUIRED]Required. The name of the workflow type.
            Note
            The combination of workflow type name and version must be unique with in a domain.
            version (string) -- [REQUIRED]Required. The version of the workflow type.
            Note
            The combination of workflow type name and version must be unique with in a domain.
            

    :type taskList: dict
    :param taskList: The task list to use for the decision tasks generated for this workflow execution. This overrides the defaultTaskList specified when registering the workflow type.
            Note
            A task list for this workflow execution must be specified either as a default for the workflow type or through this parameter. If neither this parameter is set nor a default task list was specified at registration time then a fault will be returned.
            The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
            name (string) -- [REQUIRED]The name of the task list.
            

    :type taskPriority: string
    :param taskPriority: The task priority to use for this workflow execution. This will override any default priority that was assigned when the workflow type was registered. If not set, then the default task priority for the workflow type will be used. Valid values are integers that range from Java's Integer.MIN_VALUE (-2147483648) to Integer.MAX_VALUE (2147483647). Higher numbers indicate higher priority.
            For more information about setting task priority, see Setting Task Priority in the Amazon Simple Workflow Developer Guide .
            

    :type input: string
    :param input: The input for the workflow execution. This is a free form string which should be meaningful to the workflow you are starting. This input is made available to the new workflow execution in the WorkflowExecutionStarted history event.

    :type executionStartToCloseTimeout: string
    :param executionStartToCloseTimeout: The total duration for this workflow execution. This overrides the defaultExecutionStartToCloseTimeout specified when registering the workflow type.
            The duration is specified in seconds; an integer greater than or equal to 0. Exceeding this limit will cause the workflow execution to time out. Unlike some of the other timeout parameters in Amazon SWF, you cannot specify a value of 'NONE' for this timeout; there is a one-year max limit on the time that a workflow execution can run.
            Note
            An execution start-to-close timeout must be specified either through this parameter or as a default when the workflow type is registered. If neither this parameter nor a default execution start-to-close timeout is specified, a fault is returned.
            

    :type tagList: list
    :param tagList: The list of tags to associate with the workflow execution. You can specify a maximum of 5 tags. You can list workflow executions with a specific tag by calling ListOpenWorkflowExecutions or ListClosedWorkflowExecutions and specifying a TagFilter .
            (string) --
            

    :type taskStartToCloseTimeout: string
    :param taskStartToCloseTimeout: Specifies the maximum duration of decision tasks for this workflow execution. This parameter overrides the defaultTaskStartToCloseTimout specified when registering the workflow type using RegisterWorkflowType .
            The duration is specified in seconds; an integer greater than or equal to 0. The value 'NONE' can be used to specify unlimited duration.
            Note
            A task start-to-close timeout for this workflow execution must be specified either as a default for the workflow type or through this parameter. If neither this parameter is set nor a default task start-to-close timeout was specified at registration time then a fault will be returned.
            

    :type childPolicy: string
    :param childPolicy: If set, specifies the policy to use for the child workflow executions of this workflow execution if it is terminated, by calling the TerminateWorkflowExecution action explicitly or due to an expired timeout. This policy overrides the default child policy specified when registering the workflow type using RegisterWorkflowType .
            The supported child policies are:
            TERMINATE: the child executions will be terminated.
            REQUEST_CANCEL: a request to cancel will be attempted for each child execution by recording a WorkflowExecutionCancelRequested event in its history. It is up to the decider to take appropriate actions when it receives an execution history with this event.
            ABANDON: no action will be taken. The child executions will continue to run.
            Note
            A child policy for this workflow execution must be specified either as a default for the workflow type or through this parameter. If neither this parameter is set nor a default child policy was specified at registration time then a fault will be returned.
            

    :type lambdaRole: string
    :param lambdaRole: The ARN of an IAM role that authorizes Amazon SWF to invoke AWS Lambda functions.
            Note
            In order for this workflow execution to invoke AWS Lambda functions, an appropriate IAM role must be specified either as a default for the workflow type or through this field.
            

    :rtype: dict
    :return: {
        'runId': 'string'
    }
    
    
    :returns: 
    domain (string) -- [REQUIRED]
    The name of the domain in which the workflow execution is created.
    
    workflowId (string) -- [REQUIRED]
    The user defined identifier associated with the workflow execution. You can use this to associate a custom identifier with the workflow execution. You may specify the same identifier if a workflow execution is logically a restart of a previous execution. You cannot have two open workflow executions with the same workflowId at the same time.
    The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
    
    workflowType (dict) -- [REQUIRED]
    The type of the workflow to start.
    
    name (string) -- [REQUIRED]Required. The name of the workflow type.
    
    Note
    The combination of workflow type name and version must be unique with in a domain.
    
    
    version (string) -- [REQUIRED]Required. The version of the workflow type.
    
    Note
    The combination of workflow type name and version must be unique with in a domain.
    
    
    
    
    taskList (dict) -- The task list to use for the decision tasks generated for this workflow execution. This overrides the defaultTaskList specified when registering the workflow type.
    
    Note
    A task list for this workflow execution must be specified either as a default for the workflow type or through this parameter. If neither this parameter is set nor a default task list was specified at registration time then a fault will be returned.
    
    The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
    
    name (string) -- [REQUIRED]The name of the task list.
    
    
    
    taskPriority (string) -- The task priority to use for this workflow execution. This will override any default priority that was assigned when the workflow type was registered. If not set, then the default task priority for the workflow type will be used. Valid values are integers that range from Java's Integer.MIN_VALUE (-2147483648) to Integer.MAX_VALUE (2147483647). Higher numbers indicate higher priority.
    For more information about setting task priority, see Setting Task Priority in the Amazon Simple Workflow Developer Guide .
    
    input (string) -- The input for the workflow execution. This is a free form string which should be meaningful to the workflow you are starting. This input is made available to the new workflow execution in the WorkflowExecutionStarted history event.
    executionStartToCloseTimeout (string) -- The total duration for this workflow execution. This overrides the defaultExecutionStartToCloseTimeout specified when registering the workflow type.
    The duration is specified in seconds; an integer greater than or equal to 0. Exceeding this limit will cause the workflow execution to time out. Unlike some of the other timeout parameters in Amazon SWF, you cannot specify a value of "NONE" for this timeout; there is a one-year max limit on the time that a workflow execution can run.
    
    Note
    An execution start-to-close timeout must be specified either through this parameter or as a default when the workflow type is registered. If neither this parameter nor a default execution start-to-close timeout is specified, a fault is returned.
    
    
    tagList (list) -- The list of tags to associate with the workflow execution. You can specify a maximum of 5 tags. You can list workflow executions with a specific tag by calling  ListOpenWorkflowExecutions or  ListClosedWorkflowExecutions and specifying a  TagFilter .
    
    (string) --
    
    
    taskStartToCloseTimeout (string) -- Specifies the maximum duration of decision tasks for this workflow execution. This parameter overrides the defaultTaskStartToCloseTimout specified when registering the workflow type using  RegisterWorkflowType .
    The duration is specified in seconds; an integer greater than or equal to 0. The value "NONE" can be used to specify unlimited duration.
    
    Note
    A task start-to-close timeout for this workflow execution must be specified either as a default for the workflow type or through this parameter. If neither this parameter is set nor a default task start-to-close timeout was specified at registration time then a fault will be returned.
    
    
    childPolicy (string) -- If set, specifies the policy to use for the child workflow executions of this workflow execution if it is terminated, by calling the  TerminateWorkflowExecution action explicitly or due to an expired timeout. This policy overrides the default child policy specified when registering the workflow type using  RegisterWorkflowType .
    The supported child policies are:
    
    TERMINATE: the child executions will be terminated.
    REQUEST_CANCEL: a request to cancel will be attempted for each child execution by recording a WorkflowExecutionCancelRequested event in its history. It is up to the decider to take appropriate actions when it receives an execution history with this event.
    ABANDON: no action will be taken. The child executions will continue to run.
    
    
    Note
    A child policy for this workflow execution must be specified either as a default for the workflow type or through this parameter. If neither this parameter is set nor a default child policy was specified at registration time then a fault will be returned.
    
    
    lambdaRole (string) -- The ARN of an IAM role that authorizes Amazon SWF to invoke AWS Lambda functions.
    
    Note
    In order for this workflow execution to invoke AWS Lambda functions, an appropriate IAM role must be specified either as a default for the workflow type or through this field.
    
    
    
    """
    pass