def update_deployment_group(applicationName=None, currentDeploymentGroupName=None, newDeploymentGroupName=None, deploymentConfigName=None, ec2TagFilters=None, onPremisesInstanceTagFilters=None, autoScalingGroups=None, serviceRoleArn=None, triggerConfigurations=None, alarmConfiguration=None, autoRollbackConfiguration=None, deploymentStyle=None, blueGreenDeploymentConfiguration=None, loadBalancerInfo=None):
    """
    Changes information about a deployment group.
    See also: AWS API Documentation
    
    
    :example: response = client.update_deployment_group(
        applicationName='string',
        currentDeploymentGroupName='string',
        newDeploymentGroupName='string',
        deploymentConfigName='string',
        ec2TagFilters=[
            {
                'Key': 'string',
                'Value': 'string',
                'Type': 'KEY_ONLY'|'VALUE_ONLY'|'KEY_AND_VALUE'
            },
        ],
        onPremisesInstanceTagFilters=[
            {
                'Key': 'string',
                'Value': 'string',
                'Type': 'KEY_ONLY'|'VALUE_ONLY'|'KEY_AND_VALUE'
            },
        ],
        autoScalingGroups=[
            'string',
        ],
        serviceRoleArn='string',
        triggerConfigurations=[
            {
                'triggerName': 'string',
                'triggerTargetArn': 'string',
                'triggerEvents': [
                    'DeploymentStart'|'DeploymentSuccess'|'DeploymentFailure'|'DeploymentStop'|'DeploymentRollback'|'DeploymentReady'|'InstanceStart'|'InstanceSuccess'|'InstanceFailure'|'InstanceReady',
                ]
            },
        ],
        alarmConfiguration={
            'enabled': True|False,
            'ignorePollAlarmFailure': True|False,
            'alarms': [
                {
                    'name': 'string'
                },
            ]
        },
        autoRollbackConfiguration={
            'enabled': True|False,
            'events': [
                'DEPLOYMENT_FAILURE'|'DEPLOYMENT_STOP_ON_ALARM'|'DEPLOYMENT_STOP_ON_REQUEST',
            ]
        },
        deploymentStyle={
            'deploymentType': 'IN_PLACE'|'BLUE_GREEN',
            'deploymentOption': 'WITH_TRAFFIC_CONTROL'|'WITHOUT_TRAFFIC_CONTROL'
        },
        blueGreenDeploymentConfiguration={
            'terminateBlueInstancesOnDeploymentSuccess': {
                'action': 'TERMINATE'|'KEEP_ALIVE',
                'terminationWaitTimeInMinutes': 123
            },
            'deploymentReadyOption': {
                'actionOnTimeout': 'CONTINUE_DEPLOYMENT'|'STOP_DEPLOYMENT',
                'waitTimeInMinutes': 123
            },
            'greenFleetProvisioningOption': {
                'action': 'DISCOVER_EXISTING'|'COPY_AUTO_SCALING_GROUP'
            }
        },
        loadBalancerInfo={
            'elbInfoList': [
                {
                    'name': 'string'
                },
            ]
        }
    )
    
    
    :type applicationName: string
    :param applicationName: [REQUIRED]
            The application name corresponding to the deployment group to update.
            

    :type currentDeploymentGroupName: string
    :param currentDeploymentGroupName: [REQUIRED]
            The current name of the deployment group.
            

    :type newDeploymentGroupName: string
    :param newDeploymentGroupName: The new name of the deployment group, if you want to change it.

    :type deploymentConfigName: string
    :param deploymentConfigName: The replacement deployment configuration name to use, if you want to change it.

    :type ec2TagFilters: list
    :param ec2TagFilters: The replacement set of Amazon EC2 tags on which to filter, if you want to change them. To keep the existing tags, enter their names. To remove tags, do not enter any tag names.
            (dict) --Information about an EC2 tag filter.
            Key (string) --The tag filter key.
            Value (string) --The tag filter value.
            Type (string) --The tag filter type:
            KEY_ONLY: Key only.
            VALUE_ONLY: Value only.
            KEY_AND_VALUE: Key and value.
            
            

    :type onPremisesInstanceTagFilters: list
    :param onPremisesInstanceTagFilters: The replacement set of on-premises instance tags on which to filter, if you want to change them. To keep the existing tags, enter their names. To remove tags, do not enter any tag names.
            (dict) --Information about an on-premises instance tag filter.
            Key (string) --The on-premises instance tag filter key.
            Value (string) --The on-premises instance tag filter value.
            Type (string) --The on-premises instance tag filter type:
            KEY_ONLY: Key only.
            VALUE_ONLY: Value only.
            KEY_AND_VALUE: Key and value.
            
            

    :type autoScalingGroups: list
    :param autoScalingGroups: The replacement list of Auto Scaling groups to be included in the deployment group, if you want to change them. To keep the Auto Scaling groups, enter their names. To remove Auto Scaling groups, do not enter any Auto Scaling group names.
            (string) --
            

    :type serviceRoleArn: string
    :param serviceRoleArn: A replacement ARN for the service role, if you want to change it.

    :type triggerConfigurations: list
    :param triggerConfigurations: Information about triggers to change when the deployment group is updated. For examples, see Modify Triggers in an AWS CodeDeploy Deployment Group in the AWS CodeDeploy User Guide.
            (dict) --Information about notification triggers for the deployment group.
            triggerName (string) --The name of the notification trigger.
            triggerTargetArn (string) --The ARN of the Amazon Simple Notification Service topic through which notifications about deployment or instance events are sent.
            triggerEvents (list) --The event type or types for which notifications are triggered.
            (string) --
            
            

    :type alarmConfiguration: dict
    :param alarmConfiguration: Information to add or change about Amazon CloudWatch alarms when the deployment group is updated.
            enabled (boolean) --Indicates whether the alarm configuration is enabled.
            ignorePollAlarmFailure (boolean) --Indicates whether a deployment should continue if information about the current state of alarms cannot be retrieved from Amazon CloudWatch. The default value is false.
            true: The deployment will proceed even if alarm status information can't be retrieved from Amazon CloudWatch.
            false: The deployment will stop if alarm status information can't be retrieved from Amazon CloudWatch.
            alarms (list) --A list of alarms configured for the deployment group. A maximum of 10 alarms can be added to a deployment group.
            (dict) --Information about an alarm.
            name (string) --The name of the alarm. Maximum length is 255 characters. Each alarm name can be used only once in a list of alarms.
            
            

    :type autoRollbackConfiguration: dict
    :param autoRollbackConfiguration: Information for an automatic rollback configuration that is added or changed when a deployment group is updated.
            enabled (boolean) --Indicates whether a defined automatic rollback configuration is currently enabled.
            events (list) --The event type or types that trigger a rollback.
            (string) --
            

    :type deploymentStyle: dict
    :param deploymentStyle: Information about the type of deployment, either in-place or blue/green, you want to run and whether to route deployment traffic behind a load balancer.
            deploymentType (string) --Indicates whether to run an in-place deployment or a blue/green deployment.
            deploymentOption (string) --Indicates whether to route deployment traffic behind a load balancer.
            

    :type blueGreenDeploymentConfiguration: dict
    :param blueGreenDeploymentConfiguration: Information about blue/green deployment options for a deployment group.
            terminateBlueInstancesOnDeploymentSuccess (dict) --Information about whether to terminate instances in the original fleet during a blue/green deployment.
            action (string) --The action to take on instances in the original environment after a successful blue/green deployment.
            TERMINATE: Instances are terminated after a specified wait time.
            KEEP_ALIVE: Instances are left running after they are deregistered from the load balancer and removed from the deployment group.
            terminationWaitTimeInMinutes (integer) --The number of minutes to wait after a successful blue/green deployment before terminating instances from the original environment.
            deploymentReadyOption (dict) --Information about the action to take when newly provisioned instances are ready to receive traffic in a blue/green deployment.
            actionOnTimeout (string) --Information about when to reroute traffic from an original environment to a replacement environment in a blue/green deployment.
            CONTINUE_DEPLOYMENT: Register new instances with the load balancer immediately after the new application revision is installed on the instances in the replacement environment.
            STOP_DEPLOYMENT: Do not register new instances with load balancer unless traffic is rerouted manually. If traffic is not rerouted manually before the end of the specified wait period, the deployment status is changed to Stopped.
            waitTimeInMinutes (integer) --The number of minutes to wait before the status of a blue/green deployment changed to Stopped if rerouting is not started manually. Applies only to the STOP_DEPLOYMENT option for actionOnTimeout
            greenFleetProvisioningOption (dict) --Information about how instances are provisioned for a replacement environment in a blue/green deployment.
            action (string) --The method used to add instances to a replacement environment.
            DISCOVER_EXISTING: Use instances that already exist or will be created manually.
            COPY_AUTO_SCALING_GROUP: Use settings from a specified Auto Scaling group to define and create instances in a new Auto Scaling group.
            
            

    :type loadBalancerInfo: dict
    :param loadBalancerInfo: Information about the load balancer used in a deployment.
            elbInfoList (list) --An array containing information about the load balancer in Elastic Load Balancing to use in a deployment.
            (dict) --Information about a load balancer in Elastic Load Balancing to use in a deployment.
            name (string) --For blue/green deployments, the name of the load balancer that will be used to route traffic from original instances to replacement instances in a blue/green deployment. For in-place deployments, the name of the load balancer that instances are deregistered from so they are not serving traffic during a deployment, and then re-registered with after the deployment completes.
            
            

    :rtype: dict
    :return: {
        'hooksNotCleanedUp': [
            {
                'name': 'string',
                'hook': 'string'
            },
        ]
    }
    
    
    """
    pass