def update_environment(ApplicationName=None, EnvironmentId=None, EnvironmentName=None, GroupName=None, Description=None, Tier=None, VersionLabel=None, TemplateName=None, SolutionStackName=None, PlatformArn=None, OptionSettings=None, OptionsToRemove=None):
    """
    Updates the environment description, deploys a new application version, updates the configuration settings to an entirely new configuration template, or updates select configuration option values in the running environment.
    Attempting to update both the release and configuration is not allowed and AWS Elastic Beanstalk returns an InvalidParameterCombination error.
    When updating the configuration settings to a new template or individual settings, a draft configuration is created and  DescribeConfigurationSettings for this environment returns two setting descriptions with different DeploymentStatus values.
    See also: AWS API Documentation
    
    Examples
    The following operation updates an environment named "my-env" to version "v2" of the application to which it belongs:
    Expected Output:
    The following operation configures several options in the aws:elb:loadbalancer namespace:
    Expected Output:
    
    :example: response = client.update_environment(
        ApplicationName='string',
        EnvironmentId='string',
        EnvironmentName='string',
        GroupName='string',
        Description='string',
        Tier={
            'Name': 'string',
            'Type': 'string',
            'Version': 'string'
        },
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
    :param ApplicationName: The name of the application with which the environment is associated.

    :type EnvironmentId: string
    :param EnvironmentId: The ID of the environment to update.
            If no environment with this ID exists, AWS Elastic Beanstalk returns an InvalidParameterValue error.
            Condition: You must specify either this or an EnvironmentName, or both. If you do not specify either, AWS Elastic Beanstalk returns MissingRequiredParameter error.
            

    :type EnvironmentName: string
    :param EnvironmentName: The name of the environment to update. If no environment with this name exists, AWS Elastic Beanstalk returns an InvalidParameterValue error.
            Condition: You must specify either this or an EnvironmentId, or both. If you do not specify either, AWS Elastic Beanstalk returns MissingRequiredParameter error.
            

    :type GroupName: string
    :param GroupName: The name of the group to which the target environment belongs. Specify a group name only if the environment's name is specified in an environment manifest and not with the environment name or environment ID parameters. See Environment Manifest (env.yaml) for details.

    :type Description: string
    :param Description: If this parameter is specified, AWS Elastic Beanstalk updates the description of this environment.

    :type Tier: dict
    :param Tier: This specifies the tier to use to update the environment.
            Condition: At this time, if you change the tier version, name, or type, AWS Elastic Beanstalk returns InvalidParameterValue error.
            Name (string) --The name of this environment tier.
            Type (string) --The type of this environment tier.
            Version (string) --The version of this environment tier.
            

    :type VersionLabel: string
    :param VersionLabel: If this parameter is specified, AWS Elastic Beanstalk deploys the named application version to the environment. If no such application version is found, returns an InvalidParameterValue error.

    :type TemplateName: string
    :param TemplateName: If this parameter is specified, AWS Elastic Beanstalk deploys this configuration template to the environment. If no such configuration template is found, AWS Elastic Beanstalk returns an InvalidParameterValue error.

    :type SolutionStackName: string
    :param SolutionStackName: This specifies the platform version that the environment will run after the environment is updated.

    :type PlatformArn: string
    :param PlatformArn: The ARN of the platform, if used.

    :type OptionSettings: list
    :param OptionSettings: If specified, AWS Elastic Beanstalk updates the configuration set associated with the running environment and sets the specified configuration options to the requested value.
            (dict) --A specification identifying an individual configuration option along with its current value. For a list of possible option values, go to Option Values in the AWS Elastic Beanstalk Developer Guide .
            ResourceName (string) --A unique resource name for a time-based scaling configuration option.
            Namespace (string) --A unique namespace identifying the option's associated AWS resource.
            OptionName (string) --The name of the configuration option.
            Value (string) --The current value for the configuration option.
            
            

    :type OptionsToRemove: list
    :param OptionsToRemove: A list of custom user-defined configuration options to remove from the configuration set for this environment.
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