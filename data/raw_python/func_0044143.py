def handle_event(event, context):
    """
    Do the actual event handling - try to enqueue the request.

    :param event: Lambda event that triggered the handler
    :type event: dict
    :param context: Lambda function context - see
      http://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    :return: JSON-serialized success response
    :rtype: str
    :raises: Exception
    """
    queues = queues_for_endpoint(event)
    # store some state
    msg_ids = []
    failed = 0
    # get the message to enqueue
    msg = msg_body_for_event(event, context)
    # connect to SQS API
    conn = boto3.client('sqs')
    for queue_name in queues:
        try:
            msg_ids.append(try_enqueue(conn, queue_name, msg))
        except Exception:
            failed += 1
            logger.error('Failed enqueueing message in %s:', queue_name,
                         exc_info=1)
    fail_str = ''
    status = 'success'
    if failed > 0:
        fail_str = '; %d failed' % failed
        status = 'partial'
    return {
        'status': status,
        'message': 'enqueued %s messages%s' % (len(msg_ids), fail_str),
        'SQSMessageIds': msg_ids
    }