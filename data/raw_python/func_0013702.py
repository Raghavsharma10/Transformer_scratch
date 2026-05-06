def update_stack(StackName=None, TemplateBody=None, TemplateURL=None, UsePreviousTemplate=None, StackPolicyDuringUpdateBody=None, StackPolicyDuringUpdateURL=None, Parameters=None, Capabilities=None, ResourceTypes=None, RoleARN=None, StackPolicyBody=None, StackPolicyURL=None, NotificationARNs=None, Tags=None, ClientRequestToken=None):
    """
    Updates a stack as specified in the template. After the call completes successfully, the stack update starts. You can check the status of the stack via the  DescribeStacks action.
    To get a copy of the template for an existing stack, you can use the  GetTemplate action.
    For more information about creating an update template, updating a stack, and monitoring the progress of the update, see Updating a Stack .
    See also: AWS API Documentation
    
    Examples
    This example adds two stack notification topics to the specified stack.
    Expected Output:
    
    :example: response = client.update_stack(
        StackName='string',
        TemplateBody='string',
        TemplateURL='string',
        UsePreviousTemplate=True|False,
        StackPolicyDuringUpdateBody='string',
        StackPolicyDuringUpdateURL='string',
        Parameters=[
            {
                'ParameterKey': 'string',
                'ParameterValue': 'string',
                'UsePreviousValue': True|False
            },
        ],
        Capabilities=[
            'CAPABILITY_IAM'|'CAPABILITY_NAMED_IAM',
        ],
        ResourceTypes=[
            'string',
        ],
        RoleARN='string',
        StackPolicyBody='string',
        StackPolicyURL='string',
        NotificationARNs=[
            'string',
        ],
        Tags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        ClientRequestToken='string'
    )
    
    
    :type StackName: string
    :param StackName: [REQUIRED]
            The name or unique stack ID of the stack to update.
            

    :type TemplateBody: string
    :param TemplateBody: Structure containing the template body with a minimum length of 1 byte and a maximum length of 51,200 bytes. (For more information, go to Template Anatomy in the AWS CloudFormation User Guide.)
            Conditional: You must specify only one of the following parameters: TemplateBody , TemplateURL , or set the UsePreviousTemplate to true .
            

    :type TemplateURL: string
    :param TemplateURL: Location of file containing the template body. The URL must point to a template that is located in an Amazon S3 bucket. For more information, go to Template Anatomy in the AWS CloudFormation User Guide.
            Conditional: You must specify only one of the following parameters: TemplateBody , TemplateURL , or set the UsePreviousTemplate to true .
            

    :type UsePreviousTemplate: boolean
    :param UsePreviousTemplate: Reuse the existing template that is associated with the stack that you are updating.
            Conditional: You must specify only one of the following parameters: TemplateBody , TemplateURL , or set the UsePreviousTemplate to true .
            

    :type StackPolicyDuringUpdateBody: string
    :param StackPolicyDuringUpdateBody: Structure containing the temporary overriding stack policy body. You can specify either the StackPolicyDuringUpdateBody or the StackPolicyDuringUpdateURL parameter, but not both.
            If you want to update protected resources, specify a temporary overriding stack policy during this update. If you do not specify a stack policy, the current policy that is associated with the stack will be used.
            

    :type StackPolicyDuringUpdateURL: string
    :param StackPolicyDuringUpdateURL: Location of a file containing the temporary overriding stack policy. The URL must point to a policy (max size: 16KB) located in an S3 bucket in the same region as the stack. You can specify either the StackPolicyDuringUpdateBody or the StackPolicyDuringUpdateURL parameter, but not both.
            If you want to update protected resources, specify a temporary overriding stack policy during this update. If you do not specify a stack policy, the current policy that is associated with the stack will be used.
            

    :type Parameters: list
    :param Parameters: A list of Parameter structures that specify input parameters for the stack. For more information, see the Parameter data type.
            (dict) --The Parameter data type.
            ParameterKey (string) --The key associated with the parameter. If you don't specify a key and value for a particular parameter, AWS CloudFormation uses the default value that is specified in your template.
            ParameterValue (string) --The value associated with the parameter.
            UsePreviousValue (boolean) --During a stack update, use the existing parameter value that the stack is using for a given parameter key. If you specify true , do not specify a parameter value.
            
            

    :type Capabilities: list
    :param Capabilities: A list of values that you must specify before AWS CloudFormation can update certain stacks. Some stack templates might include resources that can affect permissions in your AWS account, for example, by creating new AWS Identity and Access Management (IAM) users. For those stacks, you must explicitly acknowledge their capabilities by specifying this parameter.
            The only valid values are CAPABILITY_IAM and CAPABILITY_NAMED_IAM . The following resources require you to specify this parameter: AWS::IAM::AccessKey , AWS::IAM::Group , AWS::IAM::InstanceProfile , AWS::IAM::Policy , AWS::IAM::Role , AWS::IAM::User , and AWS::IAM::UserToGroupAddition . If your stack template contains these resources, we recommend that you review all permissions associated with them and edit their permissions if necessary.
            If you have IAM resources, you can specify either capability. If you have IAM resources with custom names, you must specify CAPABILITY_NAMED_IAM . If you don't specify this parameter, this action returns an InsufficientCapabilities error.
            For more information, see Acknowledging IAM Resources in AWS CloudFormation Templates .
            (string) --
            

    :type ResourceTypes: list
    :param ResourceTypes: The template resource types that you have permissions to work with for this update stack action, such as AWS::EC2::Instance , AWS::EC2::* , or Custom::MyCustomInstance .
            If the list of resource types doesn't include a resource that you're updating, the stack update fails. By default, AWS CloudFormation grants permissions to all resource types. AWS Identity and Access Management (IAM) uses this parameter for AWS CloudFormation-specific condition keys in IAM policies. For more information, see Controlling Access with AWS Identity and Access Management .
            (string) --
            

    :type RoleARN: string
    :param RoleARN: The Amazon Resource Name (ARN) of an AWS Identity and Access Management (IAM) role that AWS CloudFormation assumes to update the stack. AWS CloudFormation uses the role's credentials to make calls on your behalf. AWS CloudFormation always uses this role for all future operations on the stack. As long as users have permission to operate on the stack, AWS CloudFormation uses this role even if the users don't have permission to pass it. Ensure that the role grants least privilege.
            If you don't specify a value, AWS CloudFormation uses the role that was previously associated with the stack. If no role is available, AWS CloudFormation uses a temporary session that is generated from your user credentials.
            

    :type StackPolicyBody: string
    :param StackPolicyBody: Structure containing a new stack policy body. You can specify either the StackPolicyBody or the StackPolicyURL parameter, but not both.
            You might update the stack policy, for example, in order to protect a new resource that you created during a stack update. If you do not specify a stack policy, the current policy that is associated with the stack is unchanged.
            

    :type StackPolicyURL: string
    :param StackPolicyURL: Location of a file containing the updated stack policy. The URL must point to a policy (max size: 16KB) located in an S3 bucket in the same region as the stack. You can specify either the StackPolicyBody or the StackPolicyURL parameter, but not both.
            You might update the stack policy, for example, in order to protect a new resource that you created during a stack update. If you do not specify a stack policy, the current policy that is associated with the stack is unchanged.
            

    :type NotificationARNs: list
    :param NotificationARNs: Amazon Simple Notification Service topic Amazon Resource Names (ARNs) that AWS CloudFormation associates with the stack. Specify an empty list to remove all notification topics.
            (string) --
            

    :type Tags: list
    :param Tags: Key-value pairs to associate with this stack. AWS CloudFormation also propagates these tags to supported resources in the stack. You can specify a maximum number of 10 tags.
            If you don't specify this parameter, AWS CloudFormation doesn't modify the stack's tags. If you specify an empty value, AWS CloudFormation removes all associated tags.
            (dict) --The Tag type enables you to specify a key-value pair that can be used to store information about an AWS CloudFormation stack.
            Key (string) --
            Required . A string used to identify this tag. You can specify a maximum of 128 characters for a tag key. Tags owned by Amazon Web Services (AWS) have the reserved prefix: aws: .
            Value (string) --
            Required . A string containing the value for this tag. You can specify a maximum of 256 characters for a tag value.
            
            

    :type ClientRequestToken: string
    :param ClientRequestToken: A unique identifier for this UpdateStack request. Specify this token if you plan to retry requests so that AWS CloudFormation knows that you're not attempting to update a stack with the same name. You might retry UpdateStack requests to ensure that AWS CloudFormation successfully received them.

    :rtype: dict
    :return: {
        'StackId': 'string'
    }
    
    
    """
    pass