def simulate_custom_policy(PolicyInputList=None, ActionNames=None, ResourceArns=None, ResourcePolicy=None, ResourceOwner=None, CallerArn=None, ContextEntries=None, ResourceHandlingOption=None, MaxItems=None, Marker=None):
    """
    Simulate how a set of IAM policies and optionally a resource-based policy works with a list of API actions and AWS resources to determine the policies' effective permissions. The policies are provided as strings.
    The simulation does not perform the API actions; it only checks the authorization to determine if the simulated policies allow or deny the actions.
    If you want to simulate existing policies attached to an IAM user, group, or role, use  SimulatePrincipalPolicy instead.
    Context keys are variables maintained by AWS and its services that provide details about the context of an API query request. You can use the Condition element of an IAM policy to evaluate context keys. To get the list of context keys that the policies require for correct simulation, use  GetContextKeysForCustomPolicy .
    If the output is long, you can use MaxItems and Marker parameters to paginate the results.
    See also: AWS API Documentation
    
    
    :example: response = client.simulate_custom_policy(
        PolicyInputList=[
            'string',
        ],
        ActionNames=[
            'string',
        ],
        ResourceArns=[
            'string',
        ],
        ResourcePolicy='string',
        ResourceOwner='string',
        CallerArn='string',
        ContextEntries=[
            {
                'ContextKeyName': 'string',
                'ContextKeyValues': [
                    'string',
                ],
                'ContextKeyType': 'string'|'stringList'|'numeric'|'numericList'|'boolean'|'booleanList'|'ip'|'ipList'|'binary'|'binaryList'|'date'|'dateList'
            },
        ],
        ResourceHandlingOption='string',
        MaxItems=123,
        Marker='string'
    )
    
    
    :type PolicyInputList: list
    :param PolicyInputList: [REQUIRED]
            A list of policy documents to include in the simulation. Each document is specified as a string containing the complete, valid JSON text of an IAM policy. Do not include any resource-based policies in this parameter. Any resource-based policy must be submitted with the ResourcePolicy parameter. The policies cannot be 'scope-down' policies, such as you could include in a call to GetFederationToken or one of the AssumeRole APIs to restrict what a user can do while using the temporary credentials.
            The regex pattern used to validate this parameter is a string of characters consisting of any printable ASCII character ranging from the space character (u0020) through end of the ASCII character range as well as the printable characters in the Basic Latin and Latin-1 Supplement character set (through u00FF). It also includes the special characters tab (u0009), line feed (u000A), and carriage return (u000D).
            (string) --
            

    :type ActionNames: list
    :param ActionNames: [REQUIRED]
            A list of names of API actions to evaluate in the simulation. Each action is evaluated against each resource. Each action must include the service identifier, such as iam:CreateUser .
            (string) --
            

    :type ResourceArns: list
    :param ResourceArns: A list of ARNs of AWS resources to include in the simulation. If this parameter is not provided then the value defaults to * (all resources). Each API in the ActionNames parameter is evaluated for each resource in this list. The simulation determines the access result (allowed or denied) of each combination and reports it in the response.
            The simulation does not automatically retrieve policies for the specified resources. If you want to include a resource policy in the simulation, then you must include the policy as a string in the ResourcePolicy parameter.
            If you include a ResourcePolicy , then it must be applicable to all of the resources included in the simulation or you receive an invalid input error.
            For more information about ARNs, see Amazon Resource Names (ARNs) and AWS Service Namespaces in the AWS General Reference .
            (string) --
            

    :type ResourcePolicy: string
    :param ResourcePolicy: A resource-based policy to include in the simulation provided as a string. Each resource in the simulation is treated as if it had this policy attached. You can include only one resource-based policy in a simulation.
            The regex pattern used to validate this parameter is a string of characters consisting of any printable ASCII character ranging from the space character (u0020) through end of the ASCII character range as well as the printable characters in the Basic Latin and Latin-1 Supplement character set (through u00FF). It also includes the special characters tab (u0009), line feed (u000A), and carriage return (u000D).
            

    :type ResourceOwner: string
    :param ResourceOwner: An AWS account ID that specifies the owner of any simulated resource that does not identify its owner in the resource ARN, such as an S3 bucket or object. If ResourceOwner is specified, it is also used as the account owner of any ResourcePolicy included in the simulation. If the ResourceOwner parameter is not specified, then the owner of the resources and the resource policy defaults to the account of the identity provided in CallerArn . This parameter is required only if you specify a resource-based policy and account that owns the resource is different from the account that owns the simulated calling user CallerArn .

    :type CallerArn: string
    :param CallerArn: The ARN of the IAM user that you want to use as the simulated caller of the APIs. CallerArn is required if you include a ResourcePolicy so that the policy's Principal element has a value to use in evaluating the policy.
            You can specify only the ARN of an IAM user. You cannot specify the ARN of an assumed role, federated user, or a service principal.
            

    :type ContextEntries: list
    :param ContextEntries: A list of context keys and corresponding values for the simulation to use. Whenever a context key is evaluated in one of the simulated IAM permission policies, the corresponding value is supplied.
            (dict) --Contains information about a condition context key. It includes the name of the key and specifies the value (or values, if the context key supports multiple values) to use in the simulation. This information is used when evaluating the Condition elements of the input policies.
            This data type is used as an input parameter to `` SimulateCustomPolicy `` and `` SimulateCustomPolicy `` .
            ContextKeyName (string) --The full name of a condition context key, including the service prefix. For example, aws:SourceIp or s3:VersionId .
            ContextKeyValues (list) --The value (or values, if the condition context key supports multiple values) to provide to the simulation for use when the key is referenced by a Condition element in an input policy.
            (string) --
            ContextKeyType (string) --The data type of the value (or values) specified in the ContextKeyValues parameter.
            
            

    :type ResourceHandlingOption: string
    :param ResourceHandlingOption: Specifies the type of simulation to run. Different APIs that support resource-based policies require different combinations of resources. By specifying the type of simulation to run, you enable the policy simulator to enforce the presence of the required resources to ensure reliable simulation results. If your simulation does not match one of the following scenarios, then you can omit this parameter. The following list shows each of the supported scenario values and the resources that you must define to run the simulation.
            Each of the EC2 scenarios requires that you specify instance, image, and security-group resources. If your scenario includes an EBS volume, then you must specify that volume as a resource. If the EC2 scenario includes VPC, then you must supply the network-interface resource. If it includes an IP subnet, then you must specify the subnet resource. For more information on the EC2 scenario options, see Supported Platforms in the AWS EC2 User Guide .
            EC2-Classic-InstanceStore  instance, image, security-group
            EC2-Classic-EBS  instance, image, security-group, volume
            EC2-VPC-InstanceStore  instance, image, security-group, network-interface
            EC2-VPC-InstanceStore-Subnet  instance, image, security-group, network-interface, subnet
            EC2-VPC-EBS  instance, image, security-group, network-interface, volume
            EC2-VPC-EBS-Subnet  instance, image, security-group, network-interface, subnet, volume
            

    :type MaxItems: integer
    :param MaxItems: (Optional) Use this only when paginating results to indicate the maximum number of items you want in the response. If additional items exist beyond the maximum you specify, the IsTruncated response element is true .
            If you do not include this parameter, it defaults to 100. Note that IAM might return fewer results, even when there are more results available. In that case, the IsTruncated response element returns true and Marker contains a value to include in the subsequent call that tells the service where to continue from.
            

    :type Marker: string
    :param Marker: Use this parameter only when paginating results and only after you receive a response indicating that the results are truncated. Set it to the value of the Marker element in the response that you received to indicate where the next call should start.

    :rtype: dict
    :return: {
        'EvaluationResults': [
            {
                'EvalActionName': 'string',
                'EvalResourceName': 'string',
                'EvalDecision': 'allowed'|'explicitDeny'|'implicitDeny',
                'MatchedStatements': [
                    {
                        'SourcePolicyId': 'string',
                        'SourcePolicyType': 'user'|'group'|'role'|'aws-managed'|'user-managed'|'resource'|'none',
                        'StartPosition': {
                            'Line': 123,
                            'Column': 123
                        },
                        'EndPosition': {
                            'Line': 123,
                            'Column': 123
                        }
                    },
                ],
                'MissingContextValues': [
                    'string',
                ],
                'OrganizationsDecisionDetail': {
                    'AllowedByOrganizations': True|False
                },
                'EvalDecisionDetails': {
                    'string': 'allowed'|'explicitDeny'|'implicitDeny'
                },
                'ResourceSpecificResults': [
                    {
                        'EvalResourceName': 'string',
                        'EvalResourceDecision': 'allowed'|'explicitDeny'|'implicitDeny',
                        'MatchedStatements': [
                            {
                                'SourcePolicyId': 'string',
                                'SourcePolicyType': 'user'|'group'|'role'|'aws-managed'|'user-managed'|'resource'|'none',
                                'StartPosition': {
                                    'Line': 123,
                                    'Column': 123
                                },
                                'EndPosition': {
                                    'Line': 123,
                                    'Column': 123
                                }
                            },
                        ],
                        'MissingContextValues': [
                            'string',
                        ],
                        'EvalDecisionDetails': {
                            'string': 'allowed'|'explicitDeny'|'implicitDeny'
                        }
                    },
                ]
            },
        ],
        'IsTruncated': True|False,
        'Marker': 'string'
    }
    
    
    :returns: 
    (string) --
    
    """
    pass