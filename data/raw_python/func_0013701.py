def create_stack(StackName=None, TemplateBody=None, TemplateURL=None, Parameters=None, DisableRollback=None, TimeoutInMinutes=None, NotificationARNs=None, Capabilities=None, ResourceTypes=None, RoleARN=None, OnFailure=None, StackPolicyBody=None, StackPolicyURL=None, Tags=None, ClientRequestToken=None):
    """
    Creates a stack as specified in the template. After the call completes successfully, the stack creation starts. You can check the status of the stack via the  DescribeStacks API.
    See also: AWS API Documentation
    
    
    :example: response = client.create_stack(
        StackName='string',
        TemplateBody='string',
        TemplateURL='string',
        Parameters=[
            {
                'ParameterKey': 'string',
                'ParameterValue': 'string',
                'UsePreviousValue': True|False
            },
        ],
        DisableRollback=True|False,
        TimeoutInMinutes=123,
        NotificationARNs=[
            'string',
        ],
        Capabilities=[
            'CAPABILITY_IAM'|'CAPABILITY_NAMED_IAM',
        ],
        ResourceTypes=[
            'string',
        ],
        RoleARN='string',
        OnFailure='DO_NOTHING'|'ROLLBACK'|'DELETE',
        StackPolicyBody='string',
        StackPolicyURL='string',
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
            The name that is associated with the stack. The name must be unique in the region in which you are creating the stack.
            Note
            A stack name can contain only alphanumeric characters (case sensitive) and hyphens. It must start with an alphabetic character and cannot be longer than 128 characters.
            

    :type TemplateBody: string
    :param TemplateBody: Structure containing the template body with a minimum length of 1 byte and a maximum length of 51,200 bytes. For more information, go to Template Anatomy in the AWS CloudFormation User Guide.
            Conditional: You must specify either the TemplateBody or the TemplateURL parameter, but not both.
            

    :type TemplateURL: string
    :param TemplateURL: Location of file containing the template body. The URL must point to a template (max size: 460,800 bytes) that is located in an Amazon S3 bucket. For more information, go to the Template Anatomy in the AWS CloudFormation User Guide.
            Conditional: You must specify either the TemplateBody or the TemplateURL parameter, but not both.
            

    :type Parameters: list
    :param Parameters: A list of Parameter structures that specify input parameters for the stack. For more information, see the Parameter data type.
            (dict) --The Parameter data type.
            ParameterKey (string) --The key associated with the parameter. If you don't specify a key and value for a particular parameter, AWS CloudFormation uses the default value that is specified in your template.
            ParameterValue (string) --The value associated with the parameter.
            UsePreviousValue (boolean) --During a stack update, use the existing parameter value that the stack is using for a given parameter key. If you specify true , do not specify a parameter value.
            
            

    :type DisableRollback: boolean
    :param DisableRollback: Set to true to disable rollback of the stack if stack creation failed. You can specify either DisableRollback or OnFailure , but not both.
            Default: false
            

    :type TimeoutInMinutes: integer
    :param TimeoutInMinutes: The amount of time that can pass before the stack status becomes CREATE_FAILED; if DisableRollback is not set or is set to false , the stack will be rolled back.

    :type NotificationARNs: list
    :param NotificationARNs: The Simple Notification Service (SNS) topic ARNs to publish stack related events. You can find your SNS topic ARNs using the SNS console or your Command Line Interface (CLI).
            (string) --
            

    :type Capabilities: list
    :param Capabilities: A list of values that you must specify before AWS CloudFormation can create certain stacks. Some stack templates might include resources that can affect permissions in your AWS account, for example, by creating new AWS Identity and Access Management (IAM) users. For those stacks, you must explicitly acknowledge their capabilities by specifying this parameter.
            The only valid values are CAPABILITY_IAM and CAPABILITY_NAMED_IAM . The following resources require you to specify this parameter: AWS::IAM::AccessKey , AWS::IAM::Group , AWS::IAM::InstanceProfile , AWS::IAM::Policy , AWS::IAM::Role , AWS::IAM::User , and AWS::IAM::UserToGroupAddition . If your stack template contains these resources, we recommend that you review all permissions associated with them and edit their permissions if necessary.
            If you have IAM resources, you can specify either capability. If you have IAM resources with custom names, you must specify CAPABILITY_NAMED_IAM . If you don't specify this parameter, this action returns an InsufficientCapabilities error.
            For more information, see Acknowledging IAM Resources in AWS CloudFormation Templates .
            (string) --
            

    :type ResourceTypes: list
    :param ResourceTypes: The template resource types that you have permissions to work with for this create stack action, such as AWS::EC2::Instance , AWS::EC2::* , or Custom::MyCustomInstance . Use the following syntax to describe template resource types: AWS::* (for all AWS resource), Custom::* (for all custom resources), Custom::*logical_ID* `` (for a specific custom resource), ``AWS::*service_name* ::* (for all resources of a particular AWS service), and ``AWS::service_name ::resource_logical_ID `` (for a specific AWS resource).
            If the list of resource types doesn't include a resource that you're creating, the stack creation fails. By default, AWS CloudFormation grants permissions to all resource types. AWS Identity and Access Management (IAM) uses this parameter for AWS CloudFormation-specific condition keys in IAM policies. For more information, see Controlling Access with AWS Identity and Access Management .
            (string) --
            

    :type RoleARN: string
    :param RoleARN: The Amazon Resource Name (ARN) of an AWS Identity and Access Management (IAM) role that AWS CloudFormation assumes to create the stack. AWS CloudFormation uses the role's credentials to make calls on your behalf. AWS CloudFormation always uses this role for all future operations on the stack. As long as users have permission to operate on the stack, AWS CloudFormation uses this role even if the users don't have permission to pass it. Ensure that the role grants least privilege.
            If you don't specify a value, AWS CloudFormation uses the role that was previously associated with the stack. If no role is available, AWS CloudFormation uses a temporary session that is generated from your user credentials.
            

    :type OnFailure: string
    :param OnFailure: Determines what action will be taken if stack creation fails. This must be one of: DO_NOTHING, ROLLBACK, or DELETE. You can specify either OnFailure or DisableRollback , but not both.
            Default: ROLLBACK
            

    :type StackPolicyBody: string
    :param StackPolicyBody: Structure containing the stack policy body. For more information, go to Prevent Updates to Stack Resources in the AWS CloudFormation User Guide . You can specify either the StackPolicyBody or the StackPolicyURL parameter, but not both.

    :type StackPolicyURL: string
    :param StackPolicyURL: Location of a file containing the stack policy. The URL must point to a policy (maximum size: 16 KB) located in an S3 bucket in the same region as the stack. You can specify either the StackPolicyBody or the StackPolicyURL parameter, but not both.

    :type Tags: list
    :param Tags: Key-value pairs to associate with this stack. AWS CloudFormation also propagates these tags to the resources created in the stack. A maximum number of 10 tags can be specified.
            (dict) --The Tag type enables you to specify a key-value pair that can be used to store information about an AWS CloudFormation stack.
            Key (string) --
            Required . A string used to identify this tag. You can specify a maximum of 128 characters for a tag key. Tags owned by Amazon Web Services (AWS) have the reserved prefix: aws: .
            Value (string) --
            Required . A string containing the value for this tag. You can specify a maximum of 256 characters for a tag value.
            
            

    :type ClientRequestToken: string
    :param ClientRequestToken: A unique identifier for this CreateStack request. Specify this token if you plan to retry requests so that AWS CloudFormation knows that you're not attempting to create a stack with the same name. You might retry CreateStack requests to ensure that AWS CloudFormation successfully received them.

    :rtype: dict
    :return: {
        'StackId': 'string'
    }
    
    
    """
    pass