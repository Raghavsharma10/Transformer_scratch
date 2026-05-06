def try_enqueue(conn, queue_name, msg):
    """
    Try to enqueue a message. If it succeeds, return the message ID.

    :param conn: SQS API connection
    :type conn: :py:class:`botocore:SQS.Client`
    :param queue_name: name of queue to put message in
    :type queue_name: str
    :param msg: JSON-serialized message body
    :type msg: str
    :return: message ID
    :rtype: str
    """
    logger.debug('Getting Queue URL for queue %s', queue_name)
    qurl = conn.get_queue_url(QueueName=queue_name)['QueueUrl']
    logger.debug('Sending message to queue at: %s', qurl)
    resp = conn.send_message(
        QueueUrl=qurl,
        MessageBody=msg,
        DelaySeconds=0
    )
    logger.debug('Enqueued message in %s with ID %s', queue_name,
                 resp['MessageId'])
    return resp['MessageId']