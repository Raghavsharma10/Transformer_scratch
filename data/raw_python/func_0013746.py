def register_activity_type(domain=None, name=None, version=None, description=None, defaultTaskStartToCloseTimeout=None, defaultTaskHeartbeatTimeout=None, defaultTaskList=None, defaultTaskPriority=None, defaultTaskScheduleToStartTimeout=None, defaultTaskScheduleToCloseTimeout=None):
    """
    Registers a new activity type along with its configuration settings in the specified domain.
    Access Control
    You can use IAM policies to control this action's access to Amazon SWF resources as follows:
    If the caller does not have sufficient permissions to invoke the action, or the parameter values fall outside the specified constraints, the action fails. The associated event attribute's cause parameter will be set to OPERATION_NOT_PERMITTED. For details and example IAM policies, see Using IAM to Manage Access to Amazon SWF Workflows .
    See also: AWS API Documentation
    
    
    :example: response = client.register_activity_type(
        domain='string',
        name='string',
        version='string',
        description='string',
        defaultTaskStartToCloseTimeout='string',
        defaultTaskHeartbeatTimeout='string',
        defaultTaskList={
            'name': 'string'
        },
        defaultTaskPriority='string',
        defaultTaskScheduleToStartTimeout='string',
        defaultTaskScheduleToCloseTimeout='string'
    )
    
    
    :type domain: string
    :param domain: [REQUIRED]
            The name of the domain in which this activity is to be registered.
            

    :type name: string
    :param name: [REQUIRED]
            The name of the activity type within the domain.
            The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
            

    :type version: string
    :param version: [REQUIRED]
            The version of the activity type.
            Note
            The activity type consists of the name and version, the combination of which must be unique within the domain.
            The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
            

    :type description: string
    :param description: A textual description of the activity type.

    :type defaultTaskStartToCloseTimeout: string
    :param defaultTaskStartToCloseTimeout: If set, specifies the default maximum duration that a worker can take to process tasks of this activity type. This default can be overridden when scheduling an activity task using the ScheduleActivityTask decision.
            The duration is specified in seconds; an integer greater than or equal to 0. The value 'NONE' can be used to specify unlimited duration.
            

    :type defaultTaskHeartbeatTimeout: string
    :param defaultTaskHeartbeatTimeout: If set, specifies the default maximum time before which a worker processing a task of this type must report progress by calling RecordActivityTaskHeartbeat . If the timeout is exceeded, the activity task is automatically timed out. This default can be overridden when scheduling an activity task using the ScheduleActivityTask decision. If the activity worker subsequently attempts to record a heartbeat or returns a result, the activity worker receives an UnknownResource fault. In this case, Amazon SWF no longer considers the activity task to be valid; the activity worker should clean up the activity task.
            The duration is specified in seconds; an integer greater than or equal to 0. The value 'NONE' can be used to specify unlimited duration.
            

    :type defaultTaskList: dict
    :param defaultTaskList: If set, specifies the default task list to use for scheduling tasks of this activity type. This default task list is used if a task list is not provided when a task is scheduled through the ScheduleActivityTask decision.
            name (string) -- [REQUIRED]The name of the task list.
            

    :type defaultTaskPriority: string
    :param defaultTaskPriority: The default task priority to assign to the activity type. If not assigned, then '0' will be used. Valid values are integers that range from Java's Integer.MIN_VALUE (-2147483648) to Integer.MAX_VALUE (2147483647). Higher numbers indicate higher priority.
            For more information about setting task priority, see Setting Task Priority in the Amazon Simple Workflow Developer Guide .
            

    :type defaultTaskScheduleToStartTimeout: string
    :param defaultTaskScheduleToStartTimeout: If set, specifies the default maximum duration that a task of this activity type can wait before being assigned to a worker. This default can be overridden when scheduling an activity task using the ScheduleActivityTask decision.
            The duration is specified in seconds; an integer greater than or equal to 0. The value 'NONE' can be used to specify unlimited duration.
            

    :type defaultTaskScheduleToCloseTimeout: string
    :param defaultTaskScheduleToCloseTimeout: If set, specifies the default maximum duration for a task of this activity type. This default can be overridden when scheduling an activity task using the ScheduleActivityTask decision.
            The duration is specified in seconds; an integer greater than or equal to 0. The value 'NONE' can be used to specify unlimited duration.
            

    :returns: 
    domain (string) -- [REQUIRED]
    The name of the domain in which this activity is to be registered.
    
    name (string) -- [REQUIRED]
    The name of the activity type within the domain.
    The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
    
    version (string) -- [REQUIRED]
    The version of the activity type.
    
    Note
    The activity type consists of the name and version, the combination of which must be unique within the domain.
    
    The specified string must not start or end with whitespace. It must not contain a : (colon), / (slash), | (vertical bar), or any control characters (u0000-u001f | u007f - u009f). Also, it must not contain the literal string quotarnquot.
    
    description (string) -- A textual description of the activity type.
    defaultTaskStartToCloseTimeout (string) -- If set, specifies the default maximum duration that a worker can take to process tasks of this activity type. This default can be overridden when scheduling an activity task using the ScheduleActivityTask decision.
    The duration is specified in seconds; an integer greater than or equal to 0. The value "NONE" can be used to specify unlimited duration.
    
    defaultTaskHeartbeatTimeout (string) -- If set, specifies the default maximum time before which a worker processing a task of this type must report progress by calling  RecordActivityTaskHeartbeat . If the timeout is exceeded, the activity task is automatically timed out. This default can be overridden when scheduling an activity task using the ScheduleActivityTask decision. If the activity worker subsequently attempts to record a heartbeat or returns a result, the activity worker receives an UnknownResource fault. In this case, Amazon SWF no longer considers the activity task to be valid; the activity worker should clean up the activity task.
    The duration is specified in seconds; an integer greater than or equal to 0. The value "NONE" can be used to specify unlimited duration.
    
    defaultTaskList (dict) -- If set, specifies the default task list to use for scheduling tasks of this activity type. This default task list is used if a task list is not provided when a task is scheduled through the ScheduleActivityTask decision.
    
    name (string) -- [REQUIRED]The name of the task list.
    
    
    
    defaultTaskPriority (string) -- The default task priority to assign to the activity type. If not assigned, then "0" will be used. Valid values are integers that range from Java's Integer.MIN_VALUE (-2147483648) to Integer.MAX_VALUE (2147483647). Higher numbers indicate higher priority.
    For more information about setting task priority, see Setting Task Priority in the Amazon Simple Workflow Developer Guide .
    
    defaultTaskScheduleToStartTimeout (string) -- If set, specifies the default maximum duration that a task of this activity type can wait before being assigned to a worker. This default can be overridden when scheduling an activity task using the ScheduleActivityTask decision.
    The duration is specified in seconds; an integer greater than or equal to 0. The value "NONE" can be used to specify unlimited duration.
    
    defaultTaskScheduleToCloseTimeout (string) -- If set, specifies the default maximum duration for a task of this activity type. This default can be overridden when scheduling an activity task using the ScheduleActivityTask decision.
    The duration is specified in seconds; an integer greater than or equal to 0. The value "NONE" can be used to specify unlimited duration.
    
    
    """
    pass