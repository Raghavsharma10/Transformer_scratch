def update_auto_scaling_group(AutoScalingGroupName=None, LaunchConfigurationName=None, MinSize=None, MaxSize=None, DesiredCapacity=None, DefaultCooldown=None, AvailabilityZones=None, HealthCheckType=None, HealthCheckGracePeriod=None, PlacementGroup=None, VPCZoneIdentifier=None, TerminationPolicies=None, NewInstancesProtectedFromScaleIn=None):
    """
    Updates the configuration for the specified Auto Scaling group.
    The new settings take effect on any scaling activities after this call returns. Scaling activities that are currently in progress aren't affected.
    To update an Auto Scaling group with a launch configuration with InstanceMonitoring set to false , you must first disable the collection of group metrics. Otherwise, you will get an error. If you have previously enabled the collection of group metrics, you can disable it using  DisableMetricsCollection .
    Note the following:
    See also: AWS API Documentation
    
    Examples
    This example updates the launch configuration of the specified Auto Scaling group.
    Expected Output:
    This example updates the minimum size and maximum size of the specified Auto Scaling group.
    Expected Output:
    This example enables instance protection for the specified Auto Scaling group.
    Expected Output:
    
    :example: response = client.update_auto_scaling_group(
        AutoScalingGroupName='string',
        LaunchConfigurationName='string',
        MinSize=123,
        MaxSize=123,
        DesiredCapacity=123,
        DefaultCooldown=123,
        AvailabilityZones=[
            'string',
        ],
        HealthCheckType='string',
        HealthCheckGracePeriod=123,
        PlacementGroup='string',
        VPCZoneIdentifier='string',
        TerminationPolicies=[
            'string',
        ],
        NewInstancesProtectedFromScaleIn=True|False
    )
    
    
    :type AutoScalingGroupName: string
    :param AutoScalingGroupName: [REQUIRED]
            The name of the Auto Scaling group.
            

    :type LaunchConfigurationName: string
    :param LaunchConfigurationName: The name of the launch configuration.

    :type MinSize: integer
    :param MinSize: The minimum size of the Auto Scaling group.

    :type MaxSize: integer
    :param MaxSize: The maximum size of the Auto Scaling group.

    :type DesiredCapacity: integer
    :param DesiredCapacity: The number of EC2 instances that should be running in the Auto Scaling group. This number must be greater than or equal to the minimum size of the group and less than or equal to the maximum size of the group.

    :type DefaultCooldown: integer
    :param DefaultCooldown: The amount of time, in seconds, after a scaling activity completes before another scaling activity can start. The default is 300.
            For more information, see Auto Scaling Cooldowns in the Auto Scaling User Guide .
            

    :type AvailabilityZones: list
    :param AvailabilityZones: One or more Availability Zones for the group.
            (string) --
            

    :type HealthCheckType: string
    :param HealthCheckType: The service to use for the health checks. The valid values are EC2 and ELB .

    :type HealthCheckGracePeriod: integer
    :param HealthCheckGracePeriod: The amount of time, in seconds, that Auto Scaling waits before checking the health status of an EC2 instance that has come into service. The default is 0.
            For more information, see Health Checks in the Auto Scaling User Guide .
            

    :type PlacementGroup: string
    :param PlacementGroup: The name of the placement group into which you'll launch your instances, if any. For more information, see Placement Groups in the Amazon Elastic Compute Cloud User Guide .

    :type VPCZoneIdentifier: string
    :param VPCZoneIdentifier: The ID of the subnet, if you are launching into a VPC. You can specify several subnets in a comma-separated list.
            When you specify VPCZoneIdentifier with AvailabilityZones , ensure that the subnets' Availability Zones match the values you specify for AvailabilityZones .
            For more information, see Launching Auto Scaling Instances in a VPC in the Auto Scaling User Guide .
            

    :type TerminationPolicies: list
    :param TerminationPolicies: A standalone termination policy or a list of termination policies used to select the instance to terminate. The policies are executed in the order that they are listed.
            For more information, see Controlling Which Instances Auto Scaling Terminates During Scale In in the Auto Scaling User Guide .
            (string) --
            

    :type NewInstancesProtectedFromScaleIn: boolean
    :param NewInstancesProtectedFromScaleIn: Indicates whether newly launched instances are protected from termination by Auto Scaling when scaling in.

    :return: response = client.update_auto_scaling_group(
        AutoScalingGroupName='my-auto-scaling-group',
        LaunchConfigurationName='new-launch-config',
    )
    
    print(response)
    
    
    :returns: 
    AutoScalingGroupName (string) -- [REQUIRED]
    The name of the Auto Scaling group.
    
    LaunchConfigurationName (string) -- The name of the launch configuration.
    MinSize (integer) -- The minimum size of the Auto Scaling group.
    MaxSize (integer) -- The maximum size of the Auto Scaling group.
    DesiredCapacity (integer) -- The number of EC2 instances that should be running in the Auto Scaling group. This number must be greater than or equal to the minimum size of the group and less than or equal to the maximum size of the group.
    DefaultCooldown (integer) -- The amount of time, in seconds, after a scaling activity completes before another scaling activity can start. The default is 300.
    For more information, see Auto Scaling Cooldowns in the Auto Scaling User Guide .
    
    AvailabilityZones (list) -- One or more Availability Zones for the group.
    
    (string) --
    
    
    HealthCheckType (string) -- The service to use for the health checks. The valid values are EC2 and ELB .
    HealthCheckGracePeriod (integer) -- The amount of time, in seconds, that Auto Scaling waits before checking the health status of an EC2 instance that has come into service. The default is 0.
    For more information, see Health Checks in the Auto Scaling User Guide .
    
    PlacementGroup (string) -- The name of the placement group into which you'll launch your instances, if any. For more information, see Placement Groups in the Amazon Elastic Compute Cloud User Guide .
    VPCZoneIdentifier (string) -- The ID of the subnet, if you are launching into a VPC. You can specify several subnets in a comma-separated list.
    When you specify VPCZoneIdentifier with AvailabilityZones , ensure that the subnets' Availability Zones match the values you specify for AvailabilityZones .
    For more information, see Launching Auto Scaling Instances in a VPC in the Auto Scaling User Guide .
    
    TerminationPolicies (list) -- A standalone termination policy or a list of termination policies used to select the instance to terminate. The policies are executed in the order that they are listed.
    For more information, see Controlling Which Instances Auto Scaling Terminates During Scale In in the Auto Scaling User Guide .
    
    (string) --
    
    
    NewInstancesProtectedFromScaleIn (boolean) -- Indicates whether newly launched instances are protected from termination by Auto Scaling when scaling in.
    
    """
    pass