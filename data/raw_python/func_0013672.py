def put_scaling_policy(AutoScalingGroupName=None, PolicyName=None, PolicyType=None, AdjustmentType=None, MinAdjustmentStep=None, MinAdjustmentMagnitude=None, ScalingAdjustment=None, Cooldown=None, MetricAggregationType=None, StepAdjustments=None, EstimatedInstanceWarmup=None):
    """
    Creates or updates a policy for an Auto Scaling group. To update an existing policy, use the existing policy name and set the parameters you want to change. Any existing parameter not changed in an update to an existing policy is not changed in this update request.
    If you exceed your maximum limit of step adjustments, which by default is 20 per region, the call fails. For information about updating this limit, see AWS Service Limits in the Amazon Web Services General Reference .
    See also: AWS API Documentation
    
    Examples
    This example adds the specified policy to the specified Auto Scaling group.
    Expected Output:
    
    :example: response = client.put_scaling_policy(
        AutoScalingGroupName='string',
        PolicyName='string',
        PolicyType='string',
        AdjustmentType='string',
        MinAdjustmentStep=123,
        MinAdjustmentMagnitude=123,
        ScalingAdjustment=123,
        Cooldown=123,
        MetricAggregationType='string',
        StepAdjustments=[
            {
                'MetricIntervalLowerBound': 123.0,
                'MetricIntervalUpperBound': 123.0,
                'ScalingAdjustment': 123
            },
        ],
        EstimatedInstanceWarmup=123
    )
    
    
    :type AutoScalingGroupName: string
    :param AutoScalingGroupName: [REQUIRED]
            The name or ARN of the group.
            

    :type PolicyName: string
    :param PolicyName: [REQUIRED]
            The name of the policy.
            

    :type PolicyType: string
    :param PolicyType: The policy type. Valid values are SimpleScaling and StepScaling . If the policy type is null, the value is treated as SimpleScaling .

    :type AdjustmentType: string
    :param AdjustmentType: [REQUIRED]
            The adjustment type. Valid values are ChangeInCapacity , ExactCapacity , and PercentChangeInCapacity .
            For more information, see Dynamic Scaling in the Auto Scaling User Guide .
            

    :type MinAdjustmentStep: integer
    :param MinAdjustmentStep: Available for backward compatibility. Use MinAdjustmentMagnitude instead.

    :type MinAdjustmentMagnitude: integer
    :param MinAdjustmentMagnitude: The minimum number of instances to scale. If the value of AdjustmentType is PercentChangeInCapacity , the scaling policy changes the DesiredCapacity of the Auto Scaling group by at least this many instances. Otherwise, the error is ValidationError .

    :type ScalingAdjustment: integer
    :param ScalingAdjustment: The amount by which to scale, based on the specified adjustment type. A positive value adds to the current capacity while a negative number removes from the current capacity.
            This parameter is required if the policy type is SimpleScaling and not supported otherwise.
            

    :type Cooldown: integer
    :param Cooldown: The amount of time, in seconds, after a scaling activity completes and before the next scaling activity can start. If this parameter is not specified, the default cooldown period for the group applies.
            This parameter is not supported unless the policy type is SimpleScaling .
            For more information, see Auto Scaling Cooldowns in the Auto Scaling User Guide .
            

    :type MetricAggregationType: string
    :param MetricAggregationType: The aggregation type for the CloudWatch metrics. Valid values are Minimum , Maximum , and Average . If the aggregation type is null, the value is treated as Average .
            This parameter is not supported if the policy type is SimpleScaling .
            

    :type StepAdjustments: list
    :param StepAdjustments: A set of adjustments that enable you to scale based on the size of the alarm breach.
            This parameter is required if the policy type is StepScaling and not supported otherwise.
            (dict) --Describes an adjustment based on the difference between the value of the aggregated CloudWatch metric and the breach threshold that you've defined for the alarm.
            For the following examples, suppose that you have an alarm with a breach threshold of 50:
            If you want the adjustment to be triggered when the metric is greater than or equal to 50 and less than 60, specify a lower bound of 0 and an upper bound of 10.
            If you want the adjustment to be triggered when the metric is greater than 40 and less than or equal to 50, specify a lower bound of -10 and an upper bound of 0.
            There are a few rules for the step adjustments for your step policy:
            The ranges of your step adjustments can't overlap or have a gap.
            At most one step adjustment can have a null lower bound. If one step adjustment has a negative lower bound, then there must be a step adjustment with a null lower bound.
            At most one step adjustment can have a null upper bound. If one step adjustment has a positive upper bound, then there must be a step adjustment with a null upper bound.
            The upper and lower bound can't be null in the same step adjustment.
            MetricIntervalLowerBound (float) --The lower bound for the difference between the alarm threshold and the CloudWatch metric. If the metric value is above the breach threshold, the lower bound is inclusive (the metric must be greater than or equal to the threshold plus the lower bound). Otherwise, it is exclusive (the metric must be greater than the threshold plus the lower bound). A null value indicates negative infinity.
            MetricIntervalUpperBound (float) --The upper bound for the difference between the alarm threshold and the CloudWatch metric. If the metric value is above the breach threshold, the upper bound is exclusive (the metric must be less than the threshold plus the upper bound). Otherwise, it is inclusive (the metric must be less than or equal to the threshold plus the upper bound). A null value indicates positive infinity.
            The upper bound must be greater than the lower bound.
            ScalingAdjustment (integer) -- [REQUIRED]The amount by which to scale, based on the specified adjustment type. A positive value adds to the current capacity while a negative number removes from the current capacity.
            
            

    :type EstimatedInstanceWarmup: integer
    :param EstimatedInstanceWarmup: The estimated time, in seconds, until a newly launched instance can contribute to the CloudWatch metrics. The default is to use the value specified for the default cooldown period for the group.
            This parameter is not supported if the policy type is SimpleScaling .
            

    :rtype: dict
    :return: {
        'PolicyARN': 'string'
    }
    
    
    """
    pass