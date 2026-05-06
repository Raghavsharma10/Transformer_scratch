def _show_log_stream(self, conn, grp_name, stream_name, max_count=10):
        """
        Show up to ``max`` events from a specified log stream; return the
        number of events shown.

        :param conn: AWS Logs API connection
        :type conn: :py:class:`botocore:CloudWatchLogs.Client`
        :param grp_name: log group name
        :type grp_name: str
        :param stream_name: log stream name
        :type stream_name: str
        :param max_count: maximum number of events to show
        :type max_count: int
        :return: count of events shown
        :rtype: int
        """
        logger.debug('Showing up to %d events from stream %s',
                     max_count, stream_name)
        events = conn.get_log_events(
            logGroupName=grp_name,
            logStreamName=stream_name,
            limit=max_count,
            startFromHead=False
        )
        if len(events['events']) > 0:
            print('## Log Group \'%s\'; Log Stream \'%s\'' % (
                grp_name, stream_name))
        shown = 0
        for evt in events['events']:
            if shown >= max_count:
                break
            shown += 1
            dt = datetime.fromtimestamp(evt['timestamp'] / 1000.0)
            print("%s => %s" % (dt, evt['message'].strip()))
        logger.debug('displayed %d events from stream', shown)
        return shown