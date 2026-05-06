def put_scheduled_update_group_action(AutoScalingGroupName=None, ScheduledActionName=None, Time=None, StartTime=None, EndTime=None, Recurrence=None, MinSize=None, MaxSize=None, DesiredCapacity=None):
    """
    Creates or updates a scheduled scaling action for an Auto Scaling group. When updating a scheduled scaling action, if you leave a parameter unspecified, the corresponding value remains unchanged.
    For more information, see Scheduled Scaling in the Auto Scaling User Guide .
    See also: AWS API Documentation
    
    Examples
    This example adds the specified scheduled action to the specified Auto Scaling group.
    Expected Output:
    
    :example: response = client.put_scheduled_update_group_action(
        AutoScalingGroupName='string',
        ScheduledActionName='string',
        Time=datetime(2015, 1, 1),
        StartTime=datetime(2015, 1, 1),
        EndTime=datetime(2015, 1, 1),
        Recurrence='string',
        MinSize=123,
        MaxSize=123,
        DesiredCapacity=123
    )
    
    
    :type AutoScalingGroupName: string
    :param AutoScalingGroupName: [REQUIRED]
            The name or Amazon Resource Name (ARN) of the Auto Scaling group.
            

    :type ScheduledActionName: string
    :param ScheduledActionName: [REQUIRED]
            The name of this scaling action.
            

    :type Time: datetime
    :param Time: This parameter is deprecated.

    :type StartTime: datetime
    :param StartTime: The time for this action to start, in 'YYYY-MM-DDThh:mm:ssZ' format in UTC/GMT only (for example, 2014-06-01T00:00:00Z ).
            If you specify Recurrence and StartTime , Auto Scaling performs the action at this time, and then performs the action based on the specified recurrence.
            If you try to schedule your action in the past, Auto Scaling returns an error message.
            

    :type EndTime: datetime
    :param EndTime: The time for the recurring schedule to end. Auto Scaling does not perform the action after this time.

    :type Recurrence: string
    :param Recurrence: The recurring schedule for this action, in Unix cron syntax format. For more information, see Cron in Wikipedia.

    :type MinSize: integer
    :param MinSize: The minimum size for the Auto Scaling group.

    :type MaxSize: integer
    :param MaxSize: The maximum size for the Auto Scaling group.

    :type DesiredCapacity: integer
    :param DesiredCapacity: The number of EC2 instances that should be running in the group.

    :return: response = client.put_scheduled_update_group_action(
        AutoScalingGroupName='my-auto-scaling-group',
        DesiredCapacity=4,
        EndTime=datetime(2014, 5, 12, 8, 0, 0, 0, 132, 0),
        MaxSize=6,
        MinSize=2,
        ScheduledActionName='my-scheduled-action',
        StartTime=datetime(2014, 5, 12, 8, 0, 0, 0, 132, 0),
    )
    
    print(response)
    
    
    """
    pass