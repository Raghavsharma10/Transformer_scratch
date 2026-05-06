def create_target_group(Name=None, Protocol=None, Port=None, VpcId=None, HealthCheckProtocol=None, HealthCheckPort=None, HealthCheckPath=None, HealthCheckIntervalSeconds=None, HealthCheckTimeoutSeconds=None, HealthyThresholdCount=None, UnhealthyThresholdCount=None, Matcher=None):
    """
    Creates a target group.
    To register targets with the target group, use  RegisterTargets . To update the health check settings for the target group, use  ModifyTargetGroup . To monitor the health of targets in the target group, use  DescribeTargetHealth .
    To route traffic to the targets in a target group, specify the target group in an action using  CreateListener or  CreateRule .
    To delete a target group, use  DeleteTargetGroup .
    For more information, see Target Groups for Your Application Load Balancers in the Application Load Balancers Guide .
    See also: AWS API Documentation
    
    Examples
    This example creates a target group that you can use to route traffic to targets using HTTP on port 80. This target group uses the default health check configuration.
    Expected Output:
    
    :example: response = client.create_target_group(
        Name='string',
        Protocol='HTTP'|'HTTPS',
        Port=123,
        VpcId='string',
        HealthCheckProtocol='HTTP'|'HTTPS',
        HealthCheckPort='string',
        HealthCheckPath='string',
        HealthCheckIntervalSeconds=123,
        HealthCheckTimeoutSeconds=123,
        HealthyThresholdCount=123,
        UnhealthyThresholdCount=123,
        Matcher={
            'HttpCode': 'string'
        }
    )
    
    
    :type Name: string
    :param Name: [REQUIRED]
            The name of the target group.
            This name must be unique per region per account, can have a maximum of 32 characters, must contain only alphanumeric characters or hyphens, and must not begin or end with a hyphen.
            

    :type Protocol: string
    :param Protocol: [REQUIRED]
            The protocol to use for routing traffic to the targets.
            

    :type Port: integer
    :param Port: [REQUIRED]
            The port on which the targets receive traffic. This port is used unless you specify a port override when registering the target.
            

    :type VpcId: string
    :param VpcId: [REQUIRED]
            The identifier of the virtual private cloud (VPC).
            

    :type HealthCheckProtocol: string
    :param HealthCheckProtocol: The protocol the load balancer uses when performing health checks on targets. The default is the HTTP protocol.

    :type HealthCheckPort: string
    :param HealthCheckPort: The port the load balancer uses when performing health checks on targets. The default is traffic-port , which indicates the port on which each target receives traffic from the load balancer.

    :type HealthCheckPath: string
    :param HealthCheckPath: The ping path that is the destination on the targets for health checks. The default is /.

    :type HealthCheckIntervalSeconds: integer
    :param HealthCheckIntervalSeconds: The approximate amount of time, in seconds, between health checks of an individual target. The default is 30 seconds.

    :type HealthCheckTimeoutSeconds: integer
    :param HealthCheckTimeoutSeconds: The amount of time, in seconds, during which no response from a target means a failed health check. The default is 5 seconds.

    :type HealthyThresholdCount: integer
    :param HealthyThresholdCount: The number of consecutive health checks successes required before considering an unhealthy target healthy. The default is 5.

    :type UnhealthyThresholdCount: integer
    :param UnhealthyThresholdCount: The number of consecutive health check failures required before considering a target unhealthy. The default is 2.

    :type Matcher: dict
    :param Matcher: The HTTP codes to use when checking for a successful response from a target. The default is 200.
            HttpCode (string) -- [REQUIRED]The HTTP codes. You can specify values between 200 and 499. The default value is 200. You can specify multiple values (for example, '200,202') or a range of values (for example, '200-299').
            

    :rtype: dict
    :return: {
        'TargetGroups': [
            {
                'TargetGroupArn': 'string',
                'TargetGroupName': 'string',
                'Protocol': 'HTTP'|'HTTPS',
                'Port': 123,
                'VpcId': 'string',
                'HealthCheckProtocol': 'HTTP'|'HTTPS',
                'HealthCheckPort': 'string',
                'HealthCheckIntervalSeconds': 123,
                'HealthCheckTimeoutSeconds': 123,
                'HealthyThresholdCount': 123,
                'UnhealthyThresholdCount': 123,
                'HealthCheckPath': 'string',
                'Matcher': {
                    'HttpCode': 'string'
                },
                'LoadBalancerArns': [
                    'string',
                ]
            },
        ]
    }
    
    
    :returns: 
    (string) --
    
    """
    pass