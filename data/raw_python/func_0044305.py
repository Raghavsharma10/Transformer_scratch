def _delete_msg(self, conn, queue_url, receipt_handle):
        """
        Delete the message specified by ``receipt_handle`` in the queue
        specified by ``queue_url``.

        :param conn: SQS API connection
        :type conn: :py:class:`botocore:SQS.Client`
        :param queue_url: queue URL to delete the message from
        :type queue_url: str
        :param receipt_handle: message receipt handle
        :type receipt_handle: str
        """
        resp = conn.delete_message(QueueUrl=queue_url,
                                   ReceiptHandle=receipt_handle)
        if resp['ResponseMetadata']['HTTPStatusCode'] != 200:
            logger.error('Error: message with receipt handle %s in queue %s '
                         'was not successfully deleted (HTTP %s)',
                         receipt_handle, queue_url,
                         resp['ResponseMetadata']['HTTPStatusCode'])
            return
        logger.info('Message with receipt handle %s deleted from queue %s',
                    receipt_handle, queue_url)