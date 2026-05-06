def register_workflow_type(domain=None, name=None, version=None, description=None, defaultTaskStartToCloseTimeout=None, defaultExecutionStartToCloseTimeout=None, defaultTaskList=None, defaultTaskPriority=None, defaultChildPolicy=None, defaultLambdaRole=None):
    """
    Registers a new workflow type and its configuration settings in the specified domain.
    The retention period for the workflow history is set by the  RegisterDomain action.
    Access Control
    You can use IAM policies to control this action's access to Amazon SWF resources as follows:
    If the caller does not have sufficient permissions to invoke the action, or the parameter values fall outside the specified constraints, the action fails. The associated event attribute's cause parameter will be set to OPERATION_NOT_PERMITTED. For details and example IAM policies, see Using IAM to Manage Access to Amazon SWF Workflows .
    See also: AWS API Documentation
    
    
    :example: response = client.register_workflow_type(
        domain='string',
        name='string',
        version='string',
        description='string',
        defaultTaskStartToCloseTimeout='string',
        defaultExecutionStartToCloseTimeout='string',
        defaultTaskList={
            'name': 'string'
        },
        defaultTaskPriority='string',
        defaultChildPolicy='TERMINATE'|'REQUEST_CANCEL'|'ABANDON',
        defaultLambdaRole='string'
    )
    
    
    :type domain: string
    :param domain: [REQUIRED]
            The name of the domain in which to register the workflow type.
            

    :type name: string
    :param name: [REQUIRED]
            The name of the workflow type.
            The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
            

    :type version: string
    :param version: [REQUIRED]
            The version of the workflow type.
            Note
            The workflow type consists of the name and version, the combination of which must be unique within the domain. To get a list of all currently registered workflow types, use the ListWorkflowTypes action.
            The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
            

    :type description: string
    :param description: Textual description of the workflow type.

    :type defaultTaskStartToCloseTimeout: string
    :param defaultTaskStartToCloseTimeout: If set, specifies the default maximum duration of decision tasks for this workflow type. This default can be overridden when starting a workflow execution using the StartWorkflowExecution action or the StartChildWorkflowExecution decision.
            The duration is specified in seconds; an integer greater than or equal to 0. The value 'NONE' can be used to specify unlimited duration.
            

    :type defaultExecutionStartToCloseTimeout: string
    :param defaultExecutionStartToCloseTimeout: If set, specifies the default maximum duration for executions of this workflow type. You can override this default when starting an execution through the StartWorkflowExecution action or StartChildWorkflowExecution decision.
            The duration is specified in seconds; an integer greater than or equal to 0. Unlike some of the other timeout parameters in Amazon SWF, you cannot specify a value of 'NONE' for defaultExecutionStartToCloseTimeout ; there is a one-year max limit on the time that a workflow execution can run. Exceeding this limit will always cause the workflow execution to time out.
            

    :type defaultTaskList: dict
    :param defaultTaskList: If set, specifies the default task list to use for scheduling decision tasks for executions of this workflow type. This default is used only if a task list is not provided when starting the execution through the StartWorkflowExecution action or StartChildWorkflowExecution decision.
            name (string) -- [REQUIRED]The name of the task list.
            

    :type defaultTaskPriority: string
    :param defaultTaskPriority: The default task priority to assign to the workflow type. If not assigned, then '0' will be used. Valid values are integers that range from Java's Integer.MIN_VALUE (-2147483648) to Integer.MAX_VALUE (2147483647). Higher numbers indicate higher priority.
            For more information about setting task priority, see Setting Task Priority in the Amazon Simple Workflow Developer Guide .
            

    :type defaultChildPolicy: string
    :param defaultChildPolicy: If set, specifies the default policy to use for the child workflow executions when a workflow execution of this type is terminated, by calling the TerminateWorkflowExecution action explicitly or due to an expired timeout. This default can be overridden when starting a workflow execution using the StartWorkflowExecution action or the StartChildWorkflowExecution decision.
            The supported child policies are:
            TERMINATE: the child executions will be terminated.
            REQUEST_CANCEL: a request to cancel will be attempted for each child execution by recording a WorkflowExecutionCancelRequested event in its history. It is up to the decider to take appropriate actions when it receives an execution history with this event.
            ABANDON: no action will be taken. The child executions will continue to run.
            

    :type defaultLambdaRole: string
    :param defaultLambdaRole: The ARN of the default IAM role to use when a workflow execution of this type invokes AWS Lambda functions.
            This default can be overridden when starting a workflow execution using the StartWorkflowExecution action or the StartChildWorkflowExecution and ContinueAsNewWorkflowExecution decision.
            

    :returns: 
    domain (string) -- [REQUIRED]
    The name of the domain in which to register the workflow type.
    
    name (string) -- [REQUIRED]
    The name of the workflow type.
    The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
    
    version (string) -- [REQUIRED]
    The version of the workflow type.
    
    Note
    The workflow type consists of the name and version, the combination of which must be unique within the domain. To get a list of all currently registered workflow types, use the  ListWorkflowTypes action.
    
    The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
    
    description (string) -- Textual description of the workflow type.
    defaultTaskStartToCloseTimeout (string) -- If set, specifies the default maximum duration of decision tasks for this workflow type. This default can be overridden when starting a workflow execution using the  StartWorkflowExecution action or the StartChildWorkflowExecution decision.
    The duration is specified in seconds; an integer greater than or equal to 0. The value "NONE" can be used to specify unlimited duration.
    
    defaultExecutionStartToCloseTimeout (string) -- If set, specifies the default maximum duration for executions of this workflow type. You can override this default when starting an execution through the  StartWorkflowExecution action or StartChildWorkflowExecution decision.
    The duration is specified in seconds; an integer greater than or equal to 0. Unlike some of the other timeout parameters in Amazon SWF, you cannot specify a value of "NONE" for defaultExecutionStartToCloseTimeout ; there is a one-year max limit on the time that a workflow execution can run. Exceeding this limit will always cause the workflow execution to time out.
    
    defaultTaskList (dict) -- If set, specifies the default task list to use for scheduling decision tasks for executions of this workflow type. This default is used only if a task list is not provided when starting the execution through the  StartWorkflowExecution action or StartChildWorkflowExecution decision.
    
    name (string) -- [REQUIRED]The name of the task list.
    
    
    
    defaultTaskPriority (string) -- The default task priority to assign to the workflow type. If not assigned, then "0" will be used. Valid values are integers that range from Java's Integer.MIN_VALUE (-2147483648) to Integer.MAX_VALUE (2147483647). Higher numbers indicate higher priority.
    For more information about setting task priority, see Setting Task Priority in the Amazon Simple Workflow Developer Guide .
    
    defaultChildPolicy (string) -- If set, specifies the default policy to use for the child workflow executions when a workflow execution of this type is terminated, by calling the  TerminateWorkflowExecution action explicitly or due to an expired timeout. This default can be overridden when starting a workflow execution using the  StartWorkflowExecution action or the StartChildWorkflowExecution decision.
    The supported child policies are:
    
    TERMINATE: the child executions will be terminated.
    REQUEST_CANCEL: a request to cancel will be attempted for each child execution by recording a WorkflowExecutionCancelRequested event in its history. It is up to the decider to take appropriate actions when it receives an execution history with this event.
    ABANDON: no action will be taken. The child executions will continue to run.
    
    
    defaultLambdaRole (string) -- The ARN of the default IAM role to use when a workflow execution of this type invokes AWS Lambda functions.
    This default can be overridden when starting a workflow execution using the  StartWorkflowExecution action or the StartChildWorkflowExecution and ContinueAsNewWorkflowExecution decision.
    
    
    """
    pass