def create_auto_scaling_group(AutoScalingGroupName=None, LaunchConfigurationName=None, InstanceId=None, MinSize=None, MaxSize=None, DesiredCapacity=None, DefaultCooldown=None, AvailabilityZones=None, LoadBalancerNames=None, TargetGroupARNs=None, HealthCheckType=None, HealthCheckGracePeriod=None, PlacementGroup=None, VPCZoneIdentifier=None, TerminationPolicies=None, NewInstancesProtectedFromScaleIn=None, Tags=None):
    """
    Creates an Auto Scaling group with the specified name and attributes.
    If you exceed your maximum limit of Auto Scaling groups, which by default is 20 per region, the call fails. For information about viewing and updating this limit, see  DescribeAccountLimits .
    For more information, see Auto Scaling Groups in the Auto Scaling User Guide .
    See also: AWS API Documentation
    
    Examples
    This example creates an Auto Scaling group.
    Expected Output:
    This example creates an Auto Scaling group and attaches the specified Classic Load Balancer.
    Expected Output:
    This example creates an Auto Scaling group and attaches the specified target group.
    Expected Output:
    
    :example: response = client.create_auto_scaling_group(
        AutoScalingGroupName='string',
        LaunchConfigurationName='string',
        InstanceId='string',
        MinSize=123,
        MaxSize=123,
        DesiredCapacity=123,
        DefaultCooldown=123,
        AvailabilityZones=[
            'string',
        ],
        LoadBalancerNames=[
            'string',
        ],
        TargetGroupARNs=[
            'string',
        ],
        HealthCheckType='string',
        HealthCheckGracePeriod=123,
        PlacementGroup='string',
        VPCZoneIdentifier='string',
        TerminationPolicies=[
            'string',
        ],
        NewInstancesProtectedFromScaleIn=True|False,
        Tags=[
            {
                'ResourceId': 'string',
                'ResourceType': 'string',
                'Key': 'string',
                'Value': 'string',
                'PropagateAtLaunch': True|False
            },
        ]
    )
    
    
    :type AutoScalingGroupName: string
    :param AutoScalingGroupName: [REQUIRED]
            The name of the group. This name must be unique within the scope of your AWS account.
            

    :type LaunchConfigurationName: string
    :param LaunchConfigurationName: The name of the launch configuration. Alternatively, specify an EC2 instance instead of a launch configuration.

    :type InstanceId: string
    :param InstanceId: The ID of the instance used to create a launch configuration for the group. Alternatively, specify a launch configuration instead of an EC2 instance.
            When you specify an ID of an instance, Auto Scaling creates a new launch configuration and associates it with the group. This launch configuration derives its attributes from the specified instance, with the exception of the block device mapping.
            For more information, see Create an Auto Scaling Group Using an EC2 Instance in the Auto Scaling User Guide .
            

    :type MinSize: integer
    :param MinSize: [REQUIRED]
            The minimum size of the group.
            

    :type MaxSize: integer
    :param MaxSize: [REQUIRED]
            The maximum size of the group.
            

    :type DesiredCapacity: integer
    :param DesiredCapacity: The number of EC2 instances that should be running in the group. This number must be greater than or equal to the minimum size of the group and less than or equal to the maximum size of the group. If you do not specify a desired capacity, the default is the minimum size of the group.

    :type DefaultCooldown: integer
    :param DefaultCooldown: The amount of time, in seconds, after a scaling activity completes before another scaling activity can start. The default is 300.
            For more information, see Auto Scaling Cooldowns in the Auto Scaling User Guide .
            

    :type AvailabilityZones: list
    :param AvailabilityZones: One or more Availability Zones for the group. This parameter is optional if you specify one or more subnets.
            (string) --
            

    :type LoadBalancerNames: list
    :param LoadBalancerNames: One or more Classic Load Balancers. To specify an Application Load Balancer, use TargetGroupARNs instead.
            For more information, see Using a Load Balancer With an Auto Scaling Group in the Auto Scaling User Guide .
            (string) --
            

    :type TargetGroupARNs: list
    :param TargetGroupARNs: The Amazon Resource Names (ARN) of the target groups.
            (string) --
            

    :type HealthCheckType: string
    :param HealthCheckType: The service to use for the health checks. The valid values are EC2 and ELB .
            By default, health checks use Amazon EC2 instance status checks to determine the health of an instance. For more information, see Health Checks in the Auto Scaling User Guide .
            

    :type HealthCheckGracePeriod: integer
    :param HealthCheckGracePeriod: The amount of time, in seconds, that Auto Scaling waits before checking the health status of an EC2 instance that has come into service. During this time, any health check failures for the instance are ignored. The default is 0.
            This parameter is required if you are adding an ELB health check.
            For more information, see Health Checks in the Auto Scaling User Guide .
            

    :type PlacementGroup: string
    :param PlacementGroup: The name of the placement group into which you'll launch your instances, if any. For more information, see Placement Groups in the Amazon Elastic Compute Cloud User Guide .

    :type VPCZoneIdentifier: string
    :param VPCZoneIdentifier: A comma-separated list of subnet identifiers for your virtual private cloud (VPC).
            If you specify subnets and Availability Zones with this call, ensure that the subnets' Availability Zones match the Availability Zones specified.
            For more information, see Launching Auto Scaling Instances in a VPC in the Auto Scaling User Guide .
            

    :type TerminationPolicies: list
    :param TerminationPolicies: One or more termination policies used to select the instance to terminate. These policies are executed in the order that they are listed.
            For more information, see Controlling Which Instances Auto Scaling Terminates During Scale In in the Auto Scaling User Guide .
            (string) --
            

    :type NewInstancesProtectedFromScaleIn: boolean
    :param NewInstancesProtectedFromScaleIn: Indicates whether newly launched instances are protected from termination by Auto Scaling when scaling in.

    :type Tags: list
    :param Tags: One or more tags.
            For more information, see Tagging Auto Scaling Groups and Instances in the Auto Scaling User Guide .
            (dict) --Describes a tag for an Auto Scaling group.
            ResourceId (string) --The name of the group.
            ResourceType (string) --The type of resource. The only supported value is auto-scaling-group .
            Key (string) -- [REQUIRED]The tag key.
            Value (string) --The tag value.
            PropagateAtLaunch (boolean) --Determines whether the tag is added to new instances as they are launched in the group.
            
            

    :return: response = client.create_auto_scaling_group(
        AutoScalingGroupName='my-auto-scaling-group',
        LaunchConfigurationName='my-launch-config',
        MaxSize=3,
        MinSize=1,
        VPCZoneIdentifier='subnet-4176792c',
    )
    
    print(response)
    
    
    """
    pass