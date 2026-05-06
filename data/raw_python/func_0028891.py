def get_metrics(awsclient, name):
    """Print out cloudformation metrics for a lambda function.

    :param awsclient
    :param name: name of the lambda function
    :return: exit_code
    """
    metrics = ['Duration', 'Errors', 'Invocations', 'Throttles']
    client_cw = awsclient.get_client('cloudwatch')
    for metric in metrics:
        response = client_cw.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName=metric,
            Dimensions=[
                {
                    'Name': 'FunctionName',
                    'Value': name
                },
            ],
            # StartTime=datetime.now() + timedelta(days=-1),
            # EndTime=datetime.now(),
            StartTime=maya.now().subtract(days=1).datetime(),
            EndTime=maya.now().datetime(),
            Period=3600,
            Statistics=[
                'Sum',
            ],
            Unit=unit(metric)
        )
        log.info('\t%s %s' % (metric,
                           repr(aggregate_datapoints(response['Datapoints']))))
    return 0