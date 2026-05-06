def webhook2lambda2sqs_handler(event, context):
    """
    Main entry point/handler for the lambda function. Wraps
    :py:func:`~.handle_event` to ensure that we log detailed information if it
    raises an exception.

    :param event: Lambda event that triggered the handler
    :type event: dict
    :param context: Lambda function context - see
      http://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    :return: JSON-serialized success response
    :rtype: str
    :raises: Exception
    """
    # be sure we log full information about any error; if handle_event()
    # raises an exception, log a bunch of information at error level and then
    # re-raise the Exception
    try:
        res = handle_event(event, context)
    except Exception as ex:
        # log the error and re-raise the exception
        logger.error('Error handling event; event=%s context=%s',
                     event, vars(context), exc_info=1)
        raise ex
    logger.debug('handle_event() result: %s', res)
    # if all enqueues failed, this should be an error
    if len(res['SQSMessageIds']) < 1:
        raise Exception('Failed enqueueing all messages')
    # if success, return the success JSON response
    return res