def put_metric_alarm(AlarmName=None, AlarmDescription=None, ActionsEnabled=None, OKActions=None, AlarmActions=None, InsufficientDataActions=None, MetricName=None, Namespace=None, Statistic=None, ExtendedStatistic=None, Dimensions=None, Period=None, Unit=None, EvaluationPeriods=None, Threshold=None, ComparisonOperator=None, TreatMissingData=None, EvaluateLowSampleCountPercentile=None):
    """
    Creates or updates an alarm and associates it with the specified metric. Optionally, this operation can associate one or more Amazon SNS resources with the alarm.
    When this operation creates an alarm, the alarm state is immediately set to INSUFFICIENT_DATA . The alarm is evaluated and its state is set appropriately. Any actions associated with the state are then executed.
    When you update an existing alarm, its state is left unchanged, but the update completely overwrites the previous configuration of the alarm.
    If you are an AWS Identity and Access Management (IAM) user, you must have Amazon EC2 permissions for some operations:
    If you have read/write permissions for Amazon CloudWatch but not for Amazon EC2, you can still create an alarm, but the stop or terminate actions won't be performed. However, if you are later granted the required permissions, the alarm actions that you created earlier will be performed.
    If you are using an IAM role (for example, an Amazon EC2 instance profile), you cannot stop or terminate the instance using alarm actions. However, you can still see the alarm state and perform any other actions such as Amazon SNS notifications or Auto Scaling policies.
    If you are using temporary security credentials granted using the AWS Security Token Service (AWS STS), you cannot stop or terminate an Amazon EC2 instance using alarm actions.
    Note that you must create at least one stop, terminate, or reboot alarm using the Amazon EC2 or CloudWatch console to create the EC2ActionsAccess IAM role. After this IAM role is created, you can create stop, terminate, or reboot alarms using a command-line interface or an API.
    See also: AWS API Documentation
    
    
    :example: response = client.put_metric_alarm(
        AlarmName='string',
        AlarmDescription='string',
        ActionsEnabled=True|False,
        OKActions=[
            'string',
        ],
        AlarmActions=[
            'string',
        ],
        InsufficientDataActions=[
            'string',
        ],
        MetricName='string',
        Namespace='string',
        Statistic='SampleCount'|'Average'|'Sum'|'Minimum'|'Maximum',
        ExtendedStatistic='string',
        Dimensions=[
            {
                'Name': 'string',
                'Value': 'string'
            },
        ],
        Period=123,
        Unit='Seconds'|'Microseconds'|'Milliseconds'|'Bytes'|'Kilobytes'|'Megabytes'|'Gigabytes'|'Terabytes'|'Bits'|'Kilobits'|'Megabits'|'Gigabits'|'Terabits'|'Percent'|'Count'|'Bytes/Second'|'Kilobytes/Second'|'Megabytes/Second'|'Gigabytes/Second'|'Terabytes/Second'|'Bits/Second'|'Kilobits/Second'|'Megabits/Second'|'Gigabits/Second'|'Terabits/Second'|'Count/Second'|'None',
        EvaluationPeriods=123,
        Threshold=123.0,
        ComparisonOperator='GreaterThanOrEqualToThreshold'|'GreaterThanThreshold'|'LessThanThreshold'|'LessThanOrEqualToThreshold',
        TreatMissingData='string',
        EvaluateLowSampleCountPercentile='string'
    )
    
    
    :type AlarmName: string
    :param AlarmName: [REQUIRED]
            The name for the alarm. This name must be unique within the AWS account.
            

    :type AlarmDescription: string
    :param AlarmDescription: The description for the alarm.

    :type ActionsEnabled: boolean
    :param ActionsEnabled: Indicates whether actions should be executed during any changes to the alarm state.

    :type OKActions: list
    :param OKActions: The actions to execute when this alarm transitions to an OK state from any other state. Each action is specified as an Amazon Resource Name (ARN).
            Valid Values: arn:aws:automate:region :ec2:stop | arn:aws:automate:region :ec2:terminate | arn:aws:automate:region :ec2:recover
            Valid Values (for use with IAM roles): arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Stop/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Terminate/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Reboot/1.0
            (string) --
            

    :type AlarmActions: list
    :param AlarmActions: The actions to execute when this alarm transitions to the ALARM state from any other state. Each action is specified as an Amazon Resource Name (ARN).
            Valid Values: arn:aws:automate:region :ec2:stop | arn:aws:automate:region :ec2:terminate | arn:aws:automate:region :ec2:recover
            Valid Values (for use with IAM roles): arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Stop/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Terminate/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Reboot/1.0
            (string) --
            

    :type InsufficientDataActions: list
    :param InsufficientDataActions: The actions to execute when this alarm transitions to the INSUFFICIENT_DATA state from any other state. Each action is specified as an Amazon Resource Name (ARN).
            Valid Values: arn:aws:automate:region :ec2:stop | arn:aws:automate:region :ec2:terminate | arn:aws:automate:region :ec2:recover
            Valid Values (for use with IAM roles): arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Stop/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Terminate/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Reboot/1.0
            (string) --
            

    :type MetricName: string
    :param MetricName: [REQUIRED]
            The name for the metric associated with the alarm.
            

    :type Namespace: string
    :param Namespace: [REQUIRED]
            The namespace for the metric associated with the alarm.
            

    :type Statistic: string
    :param Statistic: The statistic for the metric associated with the alarm, other than percentile. For percentile statistics, use ExtendedStatistic .

    :type ExtendedStatistic: string
    :param ExtendedStatistic: The percentile statistic for the metric associated with the alarm. Specify a value between p0.0 and p100.

    :type Dimensions: list
    :param Dimensions: The dimensions for the metric associated with the alarm.
            (dict) --Expands the identity of a metric.
            Name (string) -- [REQUIRED]The name of the dimension.
            Value (string) -- [REQUIRED]The value representing the dimension measurement.
            
            

    :type Period: integer
    :param Period: [REQUIRED]
            The period, in seconds, over which the specified statistic is applied.
            

    :type Unit: string
    :param Unit: The unit of measure for the statistic. For example, the units for the Amazon EC2 NetworkIn metric are Bytes because NetworkIn tracks the number of bytes that an instance receives on all network interfaces. You can also specify a unit when you create a custom metric. Units help provide conceptual meaning to your data. Metric data points that specify a unit of measure, such as Percent, are aggregated separately.
            If you specify a unit, you must use a unit that is appropriate for the metric. Otherwise, the Amazon CloudWatch alarm can get stuck in the INSUFFICIENT DATA state.
            

    :type EvaluationPeriods: integer
    :param EvaluationPeriods: [REQUIRED]
            The number of periods over which data is compared to the specified threshold.
            

    :type Threshold: float
    :param Threshold: [REQUIRED]
            The value against which the specified statistic is compared.
            

    :type ComparisonOperator: string
    :param ComparisonOperator: [REQUIRED]
            The arithmetic operation to use when comparing the specified statistic and threshold. The specified statistic value is used as the first operand.
            

    :type TreatMissingData: string
    :param TreatMissingData: Sets how this alarm is to handle missing data points. If TreatMissingData is omitted, the default behavior of missing is used. For more information, see Configuring How CloudWatch Alarms Treats Missing Data .
            Valid Values: breaching | notBreaching | ignore | missing
            

    :type EvaluateLowSampleCountPercentile: string
    :param EvaluateLowSampleCountPercentile: Used only for alarms based on percentiles. If you specify ignore , the alarm state will not change during periods with too few data points to be statistically significant. If you specify evaluate or omit this parameter, the alarm will always be evaluated and possibly change state no matter how many data points are available. For more information, see Percentile-Based CloudWatch Alarms and Low Data Samples .
            Valid Values: evaluate | ignore
            

    :returns: 
    AlarmName (string) -- [REQUIRED]
    The name for the alarm. This name must be unique within the AWS account.
    
    AlarmDescription (string) -- The description for the alarm.
    ActionsEnabled (boolean) -- Indicates whether actions should be executed during any changes to the alarm state.
    OKActions (list) -- The actions to execute when this alarm transitions to an OK state from any other state. Each action is specified as an Amazon Resource Name (ARN).
    Valid Values: arn:aws:automate:region :ec2:stop | arn:aws:automate:region :ec2:terminate | arn:aws:automate:region :ec2:recover
    Valid Values (for use with IAM roles): arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Stop/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Terminate/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Reboot/1.0
    
    (string) --
    
    
    AlarmActions (list) -- The actions to execute when this alarm transitions to the ALARM state from any other state. Each action is specified as an Amazon Resource Name (ARN).
    Valid Values: arn:aws:automate:region :ec2:stop | arn:aws:automate:region :ec2:terminate | arn:aws:automate:region :ec2:recover
    Valid Values (for use with IAM roles): arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Stop/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Terminate/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Reboot/1.0
    
    (string) --
    
    
    InsufficientDataActions (list) -- The actions to execute when this alarm transitions to the INSUFFICIENT_DATA state from any other state. Each action is specified as an Amazon Resource Name (ARN).
    Valid Values: arn:aws:automate:region :ec2:stop | arn:aws:automate:region :ec2:terminate | arn:aws:automate:region :ec2:recover
    Valid Values (for use with IAM roles): arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Stop/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Terminate/1.0 | arn:aws:swf:us-east-1:{customer-account }:action/actions/AWS_EC2.InstanceId.Reboot/1.0
    
    (string) --
    
    
    MetricName (string) -- [REQUIRED]
    The name for the metric associated with the alarm.
    
    Namespace (string) -- [REQUIRED]
    The namespace for the metric associated with the alarm.
    
    Statistic (string) -- The statistic for the metric associated with the alarm, other than percentile. For percentile statistics, use ExtendedStatistic .
    ExtendedStatistic (string) -- The percentile statistic for the metric associated with the alarm. Specify a value between p0.0 and p100.
    Dimensions (list) -- The dimensions for the metric associated with the alarm.
    
    (dict) --Expands the identity of a metric.
    
    Name (string) -- [REQUIRED]The name of the dimension.
    
    Value (string) -- [REQUIRED]The value representing the dimension measurement.
    
    
    
    
    
    Period (integer) -- [REQUIRED]
    The period, in seconds, over which the specified statistic is applied.
    
    Unit (string) -- The unit of measure for the statistic. For example, the units for the Amazon EC2 NetworkIn metric are Bytes because NetworkIn tracks the number of bytes that an instance receives on all network interfaces. You can also specify a unit when you create a custom metric. Units help provide conceptual meaning to your data. Metric data points that specify a unit of measure, such as Percent, are aggregated separately.
    If you specify a unit, you must use a unit that is appropriate for the metric. Otherwise, the Amazon CloudWatch alarm can get stuck in the INSUFFICIENT DATA state.
    
    EvaluationPeriods (integer) -- [REQUIRED]
    The number of periods over which data is compared to the specified threshold.
    
    Threshold (float) -- [REQUIRED]
    The value against which the specified statistic is compared.
    
    ComparisonOperator (string) -- [REQUIRED]
    The arithmetic operation to use when comparing the specified statistic and threshold. The specified statistic value is used as the first operand.
    
    TreatMissingData (string) -- Sets how this alarm is to handle missing data points. If TreatMissingData is omitted, the default behavior of missing is used. For more information, see Configuring How CloudWatch Alarms Treats Missing Data .
    Valid Values: breaching | notBreaching | ignore | missing
    
    EvaluateLowSampleCountPercentile (string) -- Used only for alarms based on percentiles. If you specify ignore , the alarm state will not change during periods with too few data points to be statistically significant. If you specify evaluate or omit this parameter, the alarm will always be evaluated and possibly change state no matter how many data points are available. For more information, see Percentile-Based CloudWatch Alarms and Low Data Samples .
    Valid Values: evaluate | ignore
    
    
    """
    pass