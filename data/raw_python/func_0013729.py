def create_environment(ApplicationName=None, EnvironmentName=None, GroupName=None, Description=None, CNAMEPrefix=None, Tier=None, Tags=None, VersionLabel=None, TemplateName=None, SolutionStackName=None, PlatformArn=None, OptionSettings=None, OptionsToRemove=None):
    """
    Launches an environment for the specified application using the specified configuration.
    See also: AWS API Documentation
    
    Examples
    The following operation creates a new environment for version v1 of a java application named my-app:
    Expected Output:
    
    :example: response = client.create_environment(
        ApplicationName='string',
        EnvironmentName='string',
        GroupName='string',
        Description='string',
        CNAMEPrefix='string',
        Tier={
            'Name': 'string',
            'Type': 'string',
            'Version': 'string'
        },
        Tags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        VersionLabel='string',
        TemplateName='string',
        SolutionStackName='string',
        PlatformArn='string',
        OptionSettings=[
            {
                'ResourceName': 'string',
                'Namespace': 'string',
                'OptionName': 'string',
                'Value': 'string'
            },
        ],
        OptionsToRemove=[
            {
                'ResourceName': 'string',
                'Namespace': 'string',
                'OptionName': 'string'
            },
        ]
    )
    
    
    :type ApplicationName: string
    :param ApplicationName: [REQUIRED]
            The name of the application that contains the version to be deployed.
            If no application is found with this name, CreateEnvironment returns an InvalidParameterValue error.
            

    :type EnvironmentName: string
    :param EnvironmentName: A unique name for the deployment environment. Used in the application URL.
            Constraint: Must be from 4 to 40 characters in length. The name can contain only letters, numbers, and hyphens. It cannot start or end with a hyphen. This name must be unique in your account. If the specified name already exists, AWS Elastic Beanstalk returns an InvalidParameterValue error.
            Default: If the CNAME parameter is not specified, the environment name becomes part of the CNAME, and therefore part of the visible URL for your application.
            

    :type GroupName: string
    :param GroupName: The name of the group to which the target environment belongs. Specify a group name only if the environment's name is specified in an environment manifest and not with the environment name parameter. See Environment Manifest (env.yaml) for details.

    :type Description: string
    :param Description: Describes this environment.

    :type CNAMEPrefix: string
    :param CNAMEPrefix: If specified, the environment attempts to use this value as the prefix for the CNAME. If not specified, the CNAME is generated automatically by appending a random alphanumeric string to the environment name.

    :type Tier: dict
    :param Tier: This specifies the tier to use for creating this environment.
            Name (string) --The name of this environment tier.
            Type (string) --The type of this environment tier.
            Version (string) --The version of this environment tier.
            

    :type Tags: list
    :param Tags: This specifies the tags applied to resources in the environment.
            (dict) --Describes a tag applied to a resource in an environment.
            Key (string) --The key of the tag.
            Value (string) --The value of the tag.
            
            

    :type VersionLabel: string
    :param VersionLabel: The name of the application version to deploy.
            If the specified application has no associated application versions, AWS Elastic Beanstalk UpdateEnvironment returns an InvalidParameterValue error.
            Default: If not specified, AWS Elastic Beanstalk attempts to launch the sample application in the container.
            

    :type TemplateName: string
    :param TemplateName: The name of the configuration template to use in deployment. If no configuration template is found with this name, AWS Elastic Beanstalk returns an InvalidParameterValue error.

    :type SolutionStackName: string
    :param SolutionStackName: This is an alternative to specifying a template name. If specified, AWS Elastic Beanstalk sets the configuration values to the default values associated with the specified solution stack.

    :type PlatformArn: string
    :param PlatformArn: The ARN of the custom platform.

    :type OptionSettings: list
    :param OptionSettings: If specified, AWS Elastic Beanstalk sets the specified configuration options to the requested value in the configuration set for the new environment. These override the values obtained from the solution stack or the configuration template.
            (dict) --A specification identifying an individual configuration option along with its current value. For a list of possible option values, go to Option Values in the AWS Elastic Beanstalk Developer Guide .
            ResourceName (string) --A unique resource name for a time-based scaling configuration option.
            Namespace (string) --A unique namespace identifying the option's associated AWS resource.
            OptionName (string) --The name of the configuration option.
            Value (string) --The current value for the configuration option.
            
            

    :type OptionsToRemove: list
    :param OptionsToRemove: A list of custom user-defined configuration options to remove from the configuration set for this new environment.
            (dict) --A specification identifying an individual configuration option.
            ResourceName (string) --A unique resource name for a time-based scaling configuration option.
            Namespace (string) --A unique namespace identifying the option's associated AWS resource.
            OptionName (string) --The name of the configuration option.
            
            

    :rtype: dict
    :return: {
        'EnvironmentName': 'string',
        'EnvironmentId': 'string',
        'ApplicationName': 'string',
        'VersionLabel': 'string',
        'SolutionStackName': 'string',
        'PlatformArn': 'string',
        'TemplateName': 'string',
        'Description': 'string',
        'EndpointURL': 'string',
        'CNAME': 'string',
        'DateCreated': datetime(2015, 1, 1),
        'DateUpdated': datetime(2015, 1, 1),
        'Status': 'Launching'|'Updating'|'Ready'|'Terminating'|'Terminated',
        'AbortableOperationInProgress': True|False,
        'Health': 'Green'|'Yellow'|'Red'|'Grey',
        'HealthStatus': 'NoData'|'Unknown'|'Pending'|'Ok'|'Info'|'Warning'|'Degraded'|'Severe',
        'Resources': {
            'LoadBalancer': {
                'LoadBalancerName': 'string',
                'Domain': 'string',
                'Listeners': [
                    {
                        'Protocol': 'string',
                        'Port': 123
                    },
                ]
            }
        },
        'Tier': {
            'Name': 'string',
            'Type': 'string',
            'Version': 'string'
        },
        'EnvironmentLinks': [
            {
                'LinkName': 'string',
                'EnvironmentName': 'string'
            },
        ]
    }
    
    
    :returns: 
    Launching : Environment is in the process of initial deployment.
    Updating : Environment is in the process of updating its configuration settings or application version.
    Ready : Environment is available to have an action performed on it, such as update or terminate.
    Terminating : Environment is in the shut-down process.
    Terminated : Environment is not running.
    
    """
    pass