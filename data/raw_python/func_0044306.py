def _show_one_queue(self, conn, name, count, delete=False):
        """
        Show ``count`` messages from the specified SQS queue.

        :param conn: SQS API connection
        :type conn: :py:class:`botocore:SQS.Client`
        :param name: queue name, or None for all queues in config.
        :type name: str
        :param count: maximum number of messages to get from queue
        :type count: int
        :param delete: whether or not to delete messages after receipt
        :type delete: bool
        """
        url = self._url_for_queue(conn, name)
        logger.debug("Queue '%s' url: %s", name, url)
        logger.warning('Receiving %d messages from queue\'%s\'; this may take '
                       'up to 20 seconds.', count, name)
        if not delete:
            logger.warning("WARNING: Displayed messages will be invisible in "
                           "queue for 60 seconds!")
        seen_ids = []
        all_msgs = []
        empty_polls = 0
        # continue getting messages until we get 2 empty polls in a row
        while empty_polls < 2 and len(all_msgs) < count:
            logger.debug('Polling queue %s for messages (empty_polls=%d)',
                         name, empty_polls)
            msgs = conn.receive_message(
                QueueUrl=url,
                AttributeNames=['All'],
                MessageAttributeNames=['All'],
                MaxNumberOfMessages=count,
                VisibilityTimeout=60,
                WaitTimeSeconds=20
            )
            if 'Messages' in msgs and len(msgs['Messages']) > 0:
                empty_polls = 0
                logger.debug("Queue %s - got %d messages", name,
                             len(msgs['Messages']))
                for m in msgs['Messages']:
                    if m['MessageId'] in seen_ids:
                        continue
                    seen_ids.append(m['MessageId'])
                    all_msgs.append(m)
                continue
            # no messages found
            logger.debug('Queue %s - got no messages', name)
            empty_polls += 1
        logger.debug('received %d messages', len(all_msgs))
        if len(all_msgs) == 0:
            print('=> Queue \'%s\' appears empty.' % name)
            return
        print("=> Queue '%s' (%s)" % (name, url))
        if len(all_msgs) > count:
            all_msgs = all_msgs[:count]
        for m in all_msgs:
            try:
                m['Body'] = json.loads(m['Body'])
            except Exception:
                pass
            print(pretty_json(m))
            if delete:
                self._delete_msg(conn, url, m['ReceiptHandle'])