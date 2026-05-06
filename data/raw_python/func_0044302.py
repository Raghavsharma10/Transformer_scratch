def show_cloudwatch_logs(self, count=10, grp_name=None):
        """
        Show ``count`` latest CloudWatch Logs entries for our lambda function.

        :param count: number of log entries to show
        :type count: int
        """
        if grp_name is None:
            grp_name = '/aws/lambda/%s' % self.config.func_name
        logger.debug('Log Group Name: %s', grp_name)
        logger.debug('Connecting to AWS Logs API')
        conn = client('logs')
        logger.debug('Getting log streams')
        streams = conn.describe_log_streams(
            logGroupName=grp_name,
            orderBy='LastEventTime',
            descending=True,
            limit=count  # at worst, we have 1 event per stream
        )
        logger.debug('Found %d log streams', len(streams['logStreams']))
        shown = 0
        for stream in streams['logStreams']:
            if (count - shown) < 1:
                break
            shown += self._show_log_stream(conn, grp_name,
                                           stream['logStreamName'],
                                           (count - shown))