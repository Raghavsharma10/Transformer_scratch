def msg_body_for_event(event, context):
    """
    Generate the JSON-serialized message body for an event.

    :param event: Lambda event that triggered the handler
    :type event: dict
    :param context: Lambda function context - see
      http://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    :return: JSON-serialized success response
    :rtype: str
    """
    # find the actual input data - this differs between GET and POST
    http_method = event.get('context', {}).get('http-method', None)
    if http_method == 'GET':
        data = event.get('params', {}).get('querystring', {})
    else:  # POST
        data = event.get('body-json', {})
    # build the message to enqueue
    msg_dict = {
        'data': serializable_dict(data),
        'event': serializable_dict(event),
        'context': serializable_dict(vars(context))
    }
    msg = json.dumps(msg_dict, sort_keys=True)
    logger.debug('Message to enqueue: %s', msg)
    return msg