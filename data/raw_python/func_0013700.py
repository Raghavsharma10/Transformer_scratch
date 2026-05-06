def create_change_set(StackName=None, TemplateBody=None, TemplateURL=None, UsePreviousTemplate=None, Parameters=None, Capabilities=None, ResourceTypes=None, RoleARN=None, NotificationARNs=None, Tags=None, ChangeSetName=None, ClientToken=None, Description=None, ChangeSetType=None):
    """
    Creates a list of changes that will be applied to a stack so that you can review the changes before executing them. You can create a change set for a stack that doesn't exist or an existing stack. If you create a change set for a stack that doesn't exist, the change set shows all of the resources that AWS CloudFormation will create. If you create a change set for an existing stack, AWS CloudFormation compares the stack's information with the information that you submit in the change set and lists the differences. Use change sets to understand which resources AWS CloudFormation will create or change, and how it will change resources in an existing stack, before you create or update a stack.
    To create a change set for a stack that doesn't exist, for the ChangeSetType parameter, specify CREATE . To create a change set for an existing stack, specify UPDATE for the ChangeSetType parameter. After the CreateChangeSet call successfully completes, AWS CloudFormation starts creating the change set. To check the status of the change set or to review it, use the  DescribeChangeSet action.
    When you are satisfied with the changes the change set will make, execute the change set by using the  ExecuteChangeSet action. AWS CloudFormation doesn't make changes until you execute the change set.
    See also: AWS API Documentation
    
    
    :example: response = client.create_change_set(
        StackName='string',
        TemplateBody='string',
        TemplateURL='string',
        UsePreviousTemplate=True|False,
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
        NotificationARNs=[
            'string',
        ],
        Tags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        ChangeSetName='string',
        ClientToken='string',
        Description='string',
        ChangeSetType='CREATE'|'UPDATE'
    )
    
    
    :type StackName: string
    :param StackName: [REQUIRED]
            The name or the unique ID of the stack for which you are creating a change set. AWS CloudFormation generates the change set by comparing this stack's information with the information that you submit, such as a modified template or different parameter input values.
            

    :type TemplateBody: string
    :param TemplateBody: A structure that contains the body of the revised template, with a minimum length of 1 byte and a maximum length of 51,200 bytes. AWS CloudFormation generates the change set by comparing this template with the template of the stack that you specified.
            Conditional: You must specify only TemplateBody or TemplateURL .
            

    :type TemplateURL: string
    :param TemplateURL: The location of the file that contains the revised template. The URL must point to a template (max size: 460,800 bytes) that is located in an S3 bucket. AWS CloudFormation generates the change set by comparing this template with the stack that you specified.
            Conditional: You must specify only TemplateBody or TemplateURL .
            

    :type UsePreviousTemplate: boolean
    :param UsePreviousTemplate: Whether to reuse the template that is associated with the stack to create the change set.

    :type Parameters: list
    :param Parameters: A list of Parameter structures that specify input parameters for the change set. For more information, see the Parameter data type.
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
    :param ResourceTypes: The template resource types that you have permissions to work with if you execute this change set, such as AWS::EC2::Instance , AWS::EC2::* , or Custom::MyCustomInstance .
            If the list of resource types doesn't include a resource type that you're updating, the stack update fails. By default, AWS CloudFormation grants permissions to all resource types. AWS Identity and Access Management (IAM) uses this parameter for condition keys in IAM policies for AWS CloudFormation. For more information, see Controlling Access with AWS Identity and Access Management in the AWS CloudFormation User Guide.
            (string) --
            

    :type RoleARN: string
    :param RoleARN: The Amazon Resource Name (ARN) of an AWS Identity and Access Management (IAM) role that AWS CloudFormation assumes when executing the change set. AWS CloudFormation uses the role's credentials to make calls on your behalf. AWS CloudFormation uses this role for all future operations on the stack. As long as users have permission to operate on the stack, AWS CloudFormation uses this role even if the users don't have permission to pass it. Ensure that the role grants least privilege.
            If you don't specify a value, AWS CloudFormation uses the role that was previously associated with the stack. If no role is available, AWS CloudFormation uses a temporary session that is generated from your user credentials.
            

    :type NotificationARNs: list
    :param NotificationARNs: The Amazon Resource Names (ARNs) of Amazon Simple Notification Service (Amazon SNS) topics that AWS CloudFormation associates with the stack. To remove all associated notification topics, specify an empty list.
            (string) --
            

    :type Tags: list
    :param Tags: Key-value pairs to associate with this stack. AWS CloudFormation also propagates these tags to resources in the stack. You can specify a maximum of 10 tags.
            (dict) --The Tag type enables you to specify a key-value pair that can be used to store information about an AWS CloudFormation stack.
            Key (string) --
            Required . A string used to identify this tag. You can specify a maximum of 128 characters for a tag key. Tags owned by Amazon Web Services (AWS) have the reserved prefix: aws: .
            Value (string) --
            Required . A string containing the value for this tag. You can specify a maximum of 256 characters for a tag value.
            
            

    :type ChangeSetName: string
    :param ChangeSetName: [REQUIRED]
            The name of the change set. The name must be unique among all change sets that are associated with the specified stack.
            A change set name can contain only alphanumeric, case sensitive characters and hyphens. It must start with an alphabetic character and cannot exceed 128 characters.
            

    :type ClientToken: string
    :param ClientToken: A unique identifier for this CreateChangeSet request. Specify this token if you plan to retry requests so that AWS CloudFormation knows that you're not attempting to create another change set with the same name. You might retry CreateChangeSet requests to ensure that AWS CloudFormation successfully received them.

    :type Description: string
    :param Description: A description to help you identify this change set.

    :type ChangeSetType: string
    :param ChangeSetType: The type of change set operation. To create a change set for a new stack, specify CREATE . To create a change set for an existing stack, specify UPDATE .
            If you create a change set for a new stack, AWS Cloudformation creates a stack with a unique stack ID, but no template or resources. The stack will be in the ` REVIEW_IN_PROGRESS http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-describing-stacks.html#d0e11995`_ state until you execute the change set.
            By default, AWS CloudFormation specifies UPDATE . You can't use the UPDATE type to create a change set for a new stack or the CREATE type to create a change set for an existing stack.
            

    :rtype: dict
    :return: {
        'Id': 'string',
        'StackId': 'string'
    }
    
    
    """
    pass