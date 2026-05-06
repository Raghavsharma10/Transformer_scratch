def _get_event_source_obj(awsclient, evt_source):
    """
    Given awsclient, event_source dictionary item
    create an event_source object of the appropriate event type
    to schedule this event, and return the object.
    """
    event_source_map = {
        'dynamodb': event_source.dynamodb_stream.DynamoDBStreamEventSource,
        'kinesis': event_source.kinesis.KinesisEventSource,
        's3': event_source.s3.S3EventSource,
        'sns': event_source.sns.SNSEventSource,
        'events': event_source.cloudwatch.CloudWatchEventSource,
        'cloudfront': event_source.cloudfront.CloudFrontEventSource,
        'cloudwatch_logs': event_source.cloudwatch_logs.CloudWatchLogsEventSource,
    }

    evt_type = _get_event_type(evt_source)
    event_source_func = event_source_map.get(evt_type, None)
    if not event_source:
        raise ValueError('Unknown event source: {0}'.format(
            evt_source['arn']))

    return event_source_func(awsclient, evt_source)