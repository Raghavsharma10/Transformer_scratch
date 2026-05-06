def get_metric_statistics(Namespace=None, MetricName=None, Dimensions=None, StartTime=None, EndTime=None, Period=None, Statistics=None, ExtendedStatistics=None, Unit=None):
    """
    Gets statistics for the specified metric.
    Amazon CloudWatch retains metric data as follows:
    Note that CloudWatch started retaining 5-minute and 1-hour metric data as of 9 July 2016.
    The maximum number of data points returned from a single call is 1,440. If you request more than 1,440 data points, Amazon CloudWatch returns an error. To reduce the number of data points, you can narrow the specified time range and make multiple requests across adjacent time ranges, or you can increase the specified period. A period can be as short as one minute (60 seconds). Note that data points are not returned in chronological order.
    Amazon CloudWatch aggregates data points based on the length of the period that you specify. For example, if you request statistics with a one-hour period, Amazon CloudWatch aggregates all data points with time stamps that fall within each one-hour period. Therefore, the number of values aggregated by CloudWatch is larger than the number of data points returned.
    CloudWatch needs raw data points to calculate percentile statistics. If you publish data using a statistic set instead, you cannot retrieve percentile statistics for this data unless one of the following conditions is true:
    For a list of metrics and dimensions supported by AWS services, see the Amazon CloudWatch Metrics and Dimensions Reference in the Amazon CloudWatch User Guide .
    See also: AWS API Documentation
    
    
    :example: response = client.get_metric_statistics(
        Namespace='string',
        MetricName='string',
        Dimensions=[
            {
                'Name': 'string',
                'Value': 'string'
            },
        ],
        StartTime=datetime(2015, 1, 1),
        EndTime=datetime(2015, 1, 1),
        Period=123,
        Statistics=[
            'SampleCount'|'Average'|'Sum'|'Minimum'|'Maximum',
        ],
        ExtendedStatistics=[
            'string',
        ],
        Unit='Seconds'|'Microseconds'|'Milliseconds'|'Bytes'|'Kilobytes'|'Megabytes'|'Gigabytes'|'Terabytes'|'Bits'|'Kilobits'|'Megabits'|'Gigabits'|'Terabits'|'Percent'|'Count'|'Bytes/Second'|'Kilobytes/Second'|'Megabytes/Second'|'Gigabytes/Second'|'Terabytes/Second'|'Bits/Second'|'Kilobits/Second'|'Megabits/Second'|'Gigabits/Second'|'Terabits/Second'|'Count/Second'|'None'
    )
    
    
    :type Namespace: string
    :param Namespace: [REQUIRED]
            The namespace of the metric, with or without spaces.
            

    :type MetricName: string
    :param MetricName: [REQUIRED]
            The name of the metric, with or without spaces.
            

    :type Dimensions: list
    :param Dimensions: The dimensions. If the metric contains multiple dimensions, you must include a value for each dimension. CloudWatch treats each unique combination of dimensions as a separate metric. You can't retrieve statistics using combinations of dimensions that were not specially published. You must specify the same dimensions that were used when the metrics were created. For an example, see Dimension Combinations in the Amazon CloudWatch User Guide . For more information on specifying dimensions, see Publishing Metrics in the Amazon CloudWatch User Guide .
            (dict) --Expands the identity of a metric.
            Name (string) -- [REQUIRED]The name of the dimension.
            Value (string) -- [REQUIRED]The value representing the dimension measurement.
            
            

    :type StartTime: datetime
    :param StartTime: [REQUIRED]
            The time stamp that determines the first data point to return. Note that start times are evaluated relative to the time that CloudWatch receives the request.
            The value specified is inclusive; results include data points with the specified time stamp. The time stamp must be in ISO 8601 UTC format (for example, 2016-10-03T23:00:00Z).
            CloudWatch rounds the specified time stamp as follows:
            Start time less than 15 days ago - Round down to the nearest whole minute. For example, 12:32:34 is rounded down to 12:32:00.
            Start time between 15 and 63 days ago - Round down to the nearest 5-minute clock interval. For example, 12:32:34 is rounded down to 12:30:00.
            Start time greater than 63 days ago - Round down to the nearest 1-hour clock interval. For example, 12:32:34 is rounded down to 12:00:00.
            

    :type EndTime: datetime
    :param EndTime: [REQUIRED]
            The time stamp that determines the last data point to return.
            The value specified is exclusive; results will include data points up to the specified time stamp. The time stamp must be in ISO 8601 UTC format (for example, 2016-10-10T23:00:00Z).
            

    :type Period: integer
    :param Period: [REQUIRED]
            The granularity, in seconds, of the returned data points. A period can be as short as one minute (60 seconds) and must be a multiple of 60. The default value is 60.
            If the StartTime parameter specifies a time stamp that is greater than 15 days ago, you must specify the period as follows or no data points in that time range is returned:
            Start time between 15 and 63 days ago - Use a multiple of 300 seconds (5 minutes).
            Start time greater than 63 days ago - Use a multiple of 3600 seconds (1 hour).
            

    :type Statistics: list
    :param Statistics: The metric statistics, other than percentile. For percentile statistics, use ExtendedStatistic .
            (string) --
            

    :type ExtendedStatistics: list
    :param ExtendedStatistics: The percentile statistics. Specify values between p0.0 and p100.
            (string) --
            

    :type Unit: string
    :param Unit: The unit for a given metric. Metrics may be reported in multiple units. Not supplying a unit results in all units being returned. If the metric only ever reports one unit, specifying a unit has no effect.

    :rtype: dict
    :return: {
        'Label': 'string',
        'Datapoints': [
            {
                'Timestamp': datetime(2015, 1, 1),
                'SampleCount': 123.0,
                'Average': 123.0,
                'Sum': 123.0,
                'Minimum': 123.0,
                'Maximum': 123.0,
                'Unit': 'Seconds'|'Microseconds'|'Milliseconds'|'Bytes'|'Kilobytes'|'Megabytes'|'Gigabytes'|'Terabytes'|'Bits'|'Kilobits'|'Megabits'|'Gigabits'|'Terabits'|'Percent'|'Count'|'Bytes/Second'|'Kilobytes/Second'|'Megabytes/Second'|'Gigabytes/Second'|'Terabytes/Second'|'Bits/Second'|'Kilobits/Second'|'Megabits/Second'|'Gigabits/Second'|'Terabits/Second'|'Count/Second'|'None',
                'ExtendedStatistics': {
                    'string': 123.0
                }
            },
        ]
    }
    
    
    :returns: 
    The SampleCount of the statistic set is 1
    The Min and the Max of the statistic set are equal
    
    """
    pass