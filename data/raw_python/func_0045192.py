def _log(message):
        """
        Logs a message.

        :param str message: The log message.

        :rtype: None
        """
        #  @todo Replace with log package.
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' ' + str(message), flush=True)