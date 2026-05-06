def modify_target_group(TargetGroupArn=None, HealthCheckProtocol=None, HealthCheckPort=None, HealthCheckPath=None, HealthCheckIntervalSeconds=None, HealthCheckTimeoutSeconds=None, HealthyThresholdCount=None, UnhealthyThresholdCount=None, Matcher=None):
    """
    Modifies the health checks used when evaluating the health state of the targets in the specified target group.
    To monitor the health of the targets, use  DescribeTargetHealth .
    See also: AWS API Documentation
    
    Examples
    This example changes the configuration of the health checks used to evaluate the health of the targets for the specified target group.
    Expected Output:
    
    :example: response = client.modify_target_group(
        TargetGroupArn='string',
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
    
    
    :type TargetGroupArn: string
    :param TargetGroupArn: [REQUIRED]
            The Amazon Resource Name (ARN) of the target group.
            

    :type HealthCheckProtocol: string
    :param HealthCheckProtocol: The protocol to use to connect with the target.

    :type HealthCheckPort: string
    :param HealthCheckPort: The port to use to connect with the target.

    :type HealthCheckPath: string
    :param HealthCheckPath: The ping path that is the destination for the health check request.

    :type HealthCheckIntervalSeconds: integer
    :param HealthCheckIntervalSeconds: The approximate amount of time, in seconds, between health checks of an individual target.

    :type HealthCheckTimeoutSeconds: integer
    :param HealthCheckTimeoutSeconds: The amount of time, in seconds, during which no response means a failed health check.

    :type HealthyThresholdCount: integer
    :param HealthyThresholdCount: The number of consecutive health checks successes required before considering an unhealthy target healthy.

    :type UnhealthyThresholdCount: integer
    :param UnhealthyThresholdCount: The number of consecutive health check failures required before considering the target unhealthy.

    :type Matcher: dict
    :param Matcher: The HTTP codes to use when checking for a successful response from a target.
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