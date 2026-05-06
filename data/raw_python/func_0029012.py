def _lambda_add_s3_event_source(awsclient, arn, event, bucket, prefix,
                                suffix):
    """Use only prefix OR suffix

    :param arn:
    :param event:
    :param bucket:
    :param prefix:
    :param suffix:
    :return:
    """
    json_data = {
        'LambdaFunctionConfigurations': [{
            'LambdaFunctionArn': arn,
            'Id': str(uuid.uuid1()),
            'Events': [event]
        }]
    }

    filter_rules = build_filter_rules(prefix, suffix)

    json_data['LambdaFunctionConfigurations'][0].update({
        'Filter': {
            'Key': {
                'FilterRules': filter_rules
            }
        }
    })
    # http://docs.aws.amazon.com/cli/latest/reference/s3api/put-bucket-notification-configuration.html
    # http://docs.aws.amazon.com/AmazonS3/latest/dev/NotificationHowTo.html
    client_s3 = awsclient.get_client('s3')

    bucket_configurations = client_s3.get_bucket_notification_configuration(
        Bucket=bucket)
    bucket_configurations.pop('ResponseMetadata')

    if 'LambdaFunctionConfigurations' in bucket_configurations:
        bucket_configurations['LambdaFunctionConfigurations'].append(
            json_data['LambdaFunctionConfigurations'][0]
        )
    else:
        bucket_configurations['LambdaFunctionConfigurations'] = json_data[
            'LambdaFunctionConfigurations']

    response = client_s3.put_bucket_notification_configuration(
        Bucket=bucket,
        NotificationConfiguration=bucket_configurations
    )
    # TODO don't return a table, but success state
    return json2table(response)