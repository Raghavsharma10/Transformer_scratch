def describe_events(ApplicationName=None, VersionLabel=None, TemplateName=None, EnvironmentId=None, EnvironmentName=None, PlatformArn=None, RequestId=None, Severity=None, StartTime=None, EndTime=None, MaxRecords=None, NextToken=None):
    """
    Returns list of event descriptions matching criteria up to the last 6 weeks.
    See also: AWS API Documentation
    
    Examples
    The following operation retrieves events for an environment named my-env:
    Expected Output:
    
    :example: response = client.describe_events(
        ApplicationName='string',
        VersionLabel='string',
        TemplateName='string',
        EnvironmentId='string',
        EnvironmentName='string',
        PlatformArn='string',
        RequestId='string',
        Severity='TRACE'|'DEBUG'|'INFO'|'WARN'|'ERROR'|'FATAL',
        StartTime=datetime(2015, 1, 1),
        EndTime=datetime(2015, 1, 1),
        MaxRecords=123,
        NextToken='string'
    )
    
    
    :type ApplicationName: string
    :param ApplicationName: If specified, AWS Elastic Beanstalk restricts the returned descriptions to include only those associated with this application.

    :type VersionLabel: string
    :param VersionLabel: If specified, AWS Elastic Beanstalk restricts the returned descriptions to those associated with this application version.

    :type TemplateName: string
    :param TemplateName: If specified, AWS Elastic Beanstalk restricts the returned descriptions to those that are associated with this environment configuration.

    :type EnvironmentId: string
    :param EnvironmentId: If specified, AWS Elastic Beanstalk restricts the returned descriptions to those associated with this environment.

    :type EnvironmentName: string
    :param EnvironmentName: If specified, AWS Elastic Beanstalk restricts the returned descriptions to those associated with this environment.

    :type PlatformArn: string
    :param PlatformArn: The ARN of the version of the custom platform.

    :type RequestId: string
    :param RequestId: If specified, AWS Elastic Beanstalk restricts the described events to include only those associated with this request ID.

    :type Severity: string
    :param Severity: If specified, limits the events returned from this call to include only those with the specified severity or higher.

    :type StartTime: datetime
    :param StartTime: If specified, AWS Elastic Beanstalk restricts the returned descriptions to those that occur on or after this time.

    :type EndTime: datetime
    :param EndTime: If specified, AWS Elastic Beanstalk restricts the returned descriptions to those that occur up to, but not including, the EndTime .

    :type MaxRecords: integer
    :param MaxRecords: Specifies the maximum number of events that can be returned, beginning with the most recent event.

    :type NextToken: string
    :param NextToken: Pagination token. If specified, the events return the next batch of results.

    :rtype: dict
    :return: {
        'Events': [
            {
                'EventDate': datetime(2015, 1, 1),
                'Message': 'string',
                'ApplicationName': 'string',
                'VersionLabel': 'string',
                'TemplateName': 'string',
                'EnvironmentName': 'string',
                'PlatformArn': 'string',
                'RequestId': 'string',
                'Severity': 'TRACE'|'DEBUG'|'INFO'|'WARN'|'ERROR'|'FATAL'
            },
        ],
        'NextToken': 'string'
    }
    
    
    """
    pass